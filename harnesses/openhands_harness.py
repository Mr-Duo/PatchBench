"""Runs OpenHands through the OpenHands SDK, configured via a YAML file."""

import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import yaml
from openhands.sdk import LLM, Agent, Conversation
from openhands.sdk.event.llm_convertible.observation import ObservationEvent
from openhands.sdk.tool import Tool, ToolExecutor, register_tool
from openhands.tools.file_editor import FileEditorTool as StockFileEditorTool
from openhands.tools.file_editor.definition import FileEditorObservation
from openhands.tools.preset.default import get_default_tools
from openhands.tools.terminal import TerminalTool

from agent_base import AGENT_HOME, API_KEY_NAMES, AgentHarness, main
from constants import MODEL_PRICING

#: Benchmark model names mapped onto the names litellm expects. Only applied
#: when the YAML's model_name matches a key; otherwise it is used as written.
MODEL_MAPPINGS = {
    'gpt-4.1-2025-04-14': 'gpt-4.1',
    'o4-mini-2025-04-16': 'o4-mini',
    'o3-2025-04-16': 'o3',
    'gemini-3.1-pro-preview': 'gemini/gemini-3.1-pro-preview',
    'gemini-3.5-flash': 'gemini/gemini-3.5-flash',
    'kimi-k2.6': 'openai/kimi-k2.6',
    'gpt-5.6-sol': 'openai/gpt-5.6-sol',
}

#: file_editor commands that write, as opposed to reading or listing.
WRITE_COMMANDS = {"create", "str_replace", "insert", "undo_edit"}
#: Where the conversation is saved; copied to the log directory afterwards.
CONVERSATIONS_DIR = Path(AGENT_HOME, ".openhands", "conversations")
#: How long the conversation gets to exit when stopped at the budget.
STOP_GRACE_SECONDS = 10
#: Env var that points at the YAML config.
CONFIG_ENV = "OPENHANDS_CONFIG"
DEFAULT_CONFIG = "openhands_config.yaml"


def load_config(path: str | os.PathLike) -> dict:
    """Read and validate the YAML config."""
    with open(path) as handle:
        cfg = yaml.safe_load(handle) or {}

    if not cfg.get("model_name"):
        raise ValueError(f"{path}: 'model_name' is required.")
    if not cfg.get("api_key_name") and not cfg.get("auth_token_name"):
        raise ValueError(f"{path}: set at least one of 'api_key_name' or 'auth_token_name'.")
    if cfg["model_name"] not in MODEL_PRICING:
        raise ValueError(f"{path}: no MODEL_PRICING entry for model '{cfg['model_name']}'.")

    for field in ("base_url", "api_key_name", "auth_token_name", "effort_level"):
        cfg.setdefault(field, None)
    return cfg


def read_secret(env_name: str | None) -> str | None:
    """Return the value of the env var called ``env_name`` (None if no name given)."""
    if not env_name:
        return None
    value = os.getenv(env_name)
    if not value:
        raise RuntimeError(f"Environment variable {env_name} is not set.")
    return value


class OpenhandsHarness(AgentHarness):
    name = "openhands"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = load_config(os.getenv(CONFIG_ENV, DEFAULT_CONFIG))
        # The YAML is the source of truth for the model.
        self.model_name = self.config["model_name"]
        # Fail before starting if a configured secret is missing.
        read_secret(self.config["api_key_name"])
        read_secret(self.config["auth_token_name"])
        self._changed: set[str] = set()

    # -- running the agent --------------------------------------------------

    def invoke(self, prompt: str) -> None:
        """Run the conversation in a separate process, and stop it at the budget.

        OpenHands runs its tools in whichever Python process runs the
        conversation, and the tools must run as the agent's user, so the
        conversation gets a process of its own (``run_conversation``).
        """
        report_r, report_w = os.pipe()
        keys = {name: os.environ[name] for name in API_KEY_NAMES if name in os.environ}
        # The configured secrets, even if their names are not in API_KEY_NAMES.
        for field in ("api_key_name", "auth_token_name"):
            name = self.config[field]
            if name:
                keys[name] = os.environ[name]
        env = self.agent_env(PYTHONPATH=os.environ.get("PYTHONPATH", ""),
                             PATCHBENCH_REPORT_FD=str(report_w), **keys)
        conversation = subprocess.Popen([sys.executable, __file__, "--conversation"],
                                        stdin=subprocess.PIPE, text=True, env=env,
                                        pass_fds=(report_w,), **self.as_agent())
        os.close(report_w)
        conversation.stdin.write(json.dumps({"config": self.config,
                                             "repo_folder": self.repo_folder, "prompt": prompt}))
        conversation.stdin.close()

        spend = None
        try:
            with os.fdopen(report_r) as reports:
                for line in reports:
                    report = json.loads(line)
                    if "changed" in report:
                        self._changed.add(report["changed"])
                    elif report["spend"] != spend:
                        spend = report["spend"]
                        if self.note_spend(spend):
                            break
        finally:
            if conversation.poll() is None:
                conversation.send_signal(signal.SIGTERM)
            try:
                conversation.wait(timeout=STOP_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                conversation.kill()
                conversation.wait()
            if CONVERSATIONS_DIR.is_dir():
                shutil.copytree(CONVERSATIONS_DIR, self.log_dir / "conversations", dirs_exist_ok=True)

    # -- changed files ------------------------------------------------------

    def changed_files(self) -> list[str]:
        """Paths written by successful ``file_editor`` commands, as the conversation reported."""
        return sorted(self._changed)


class Gate(ToolExecutor):
    """Refuses file_editor calls outside the folders shared with the task container."""

    def __init__(self, inner, cwd: str):
        self.inner = inner
        self.cwd = cwd

    def __call__(self, action, conversation=None):
        if AgentHarness.file_allowed(action.path, self.cwd):
            return self.inner(action, conversation)
        return FileEditorObservation.from_text(text=f"The path {action.path} does not exist.",
                                               command=action.command, is_error=True)


class FileEditorTool(StockFileEditorTool):
    """The stock file editor behind :class:`Gate`.

    OpenHands names a tool after its class, so the model still sees ``file_editor``.
    """

    @classmethod
    def create(cls, conv_state, **kwargs):
        cwd = str(conv_state.workspace.working_dir)
        return [tool.set_executor(Gate(tool.executor, cwd))
                for tool in super().create(conv_state, **kwargs)]


def run_conversation() -> None:
    """The conversation itself, run as the agent's user (see ``OpenhandsHarness.invoke``).

    Reads the config, checkout and prompt as JSON on stdin. After every event it
    reports the spend so far, and each file a ``file_editor`` command wrote, one
    JSON object per line on ``PATCHBENCH_REPORT_FD``.
    """
    job = json.load(sys.stdin)
    cfg = job["config"]
    reports = os.fdopen(int(os.environ["PATCHBENCH_REPORT_FD"]), "w", buffering=1)
    pricing = MODEL_PRICING[cfg["model_name"]]

    api_key = read_secret(cfg["api_key_name"])
    auth_token = read_secret(cfg["auth_token_name"])
    # Both credentials end up as a Bearer header; with only one configured it
    # serves as the api_key.
    key = api_key or auth_token
    # litellm reads the key from the environment as well as the config.
    os.environ["LLM_API_KEY"] = key
    llm_kwargs = {"model": MODEL_MAPPINGS.get(cfg["model_name"], cfg["model_name"]),
                  "api_key": key}
    if api_key and auth_token:
        llm_kwargs["extra_headers"] = {"Authorization": f"Bearer {auth_token}"}
    if cfg["base_url"]:
        llm_kwargs["base_url"] = cfg["base_url"]
    if cfg["effort_level"]:
        llm_kwargs["reasoning_effort"] = cfg["effort_level"]
    llm = LLM(**llm_kwargs)

    tools = get_default_tools(enable_browser=False)
    register_tool(FileEditorTool.name, FileEditorTool)  # replaces the stock file_editor
    tools = [Tool(name=tool.name, params={"terminal_type": "subprocess"})
             if tool.name == TerminalTool.name else tool for tool in tools]

    conversation = None

    def spend() -> float:
        usage = (conversation.conversation_stats.get_combined_metrics().accumulated_token_usage
                 if conversation is not None else None)
        if usage is None:
            return 0.0
        uncached = max(usage.prompt_tokens - usage.cache_read_tokens, 0)
        return (uncached * pricing["input"]
                + usage.cache_read_tokens * pricing["cached_input"]
                + usage.cache_write_tokens * pricing.get("cache_write", 0)
                + usage.completion_tokens * pricing["output"]) / 1_000_000

    def report(event) -> None:
        if isinstance(event, ObservationEvent) and event.tool_name == "file_editor":
            obs = event.observation.model_dump()
            if not obs.get("is_error") and obs.get("command") in WRITE_COMMANDS and obs.get("path"):
                reports.write(json.dumps({"changed": obs["path"]}) + "\n")
        reports.write(json.dumps({"spend": spend()}) + "\n")

    conversation = Conversation(agent=Agent(llm=llm, tools=tools), workspace=job["repo_folder"],
                                persistence_dir=str(CONVERSATIONS_DIR), callbacks=[report])
    conversation.send_message(job["prompt"])
    try:
        conversation.run()
    finally:
        conversation.close()


if __name__ == "__main__":
    if sys.argv[1:] == ["--conversation"]:
        run_conversation()
    else:
        main(OpenhandsHarness)