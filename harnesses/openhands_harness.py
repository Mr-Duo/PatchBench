"""Runs OpenHands through the OpenHands SDK, via the OpenRouter API."""

import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from openhands.sdk import LLM, Agent, Conversation
from openhands.sdk.event.llm_convertible.observation import ObservationEvent
from openhands.sdk.tool import Tool, ToolExecutor, register_tool
from openhands.tools.file_editor import FileEditorTool as StockFileEditorTool
from openhands.tools.file_editor.definition import FileEditorObservation
from openhands.tools.preset.default import get_default_tools
from openhands.tools.terminal import TerminalTool

from agent_base import AGENT_HOME, AgentHarness, main
from constants import MODEL_PRICING

#: OpenRouter's OpenAI-compatible endpoint.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
#: Env var holding the OpenRouter key.
OPENROUTER_KEY_NAME = "OPENROUTER_API_KEY"

#: file_editor commands that write, as opposed to reading or listing.
WRITE_COMMANDS = {"create", "str_replace", "insert", "undo_edit"}
#: Where the conversation is saved; copied to the log directory afterwards.
CONVERSATIONS_DIR = Path(AGENT_HOME, ".openhands", "conversations")
#: How long the conversation gets to exit when stopped at the budget.
STOP_GRACE_SECONDS = 10


def _litellm_model(model_name: str) -> str:
    """The litellm name for an OpenRouter slug, e.g. ``anthropic/claude-sonnet-4.5``."""
    return model_name if model_name.startswith("openrouter/") else f"openrouter/{model_name}"


class OpenhandsHarness(AgentHarness):
    name = "openhands"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.model_name must be an OpenRouter slug, e.g. "anthropic/claude-sonnet-4.5",
        # and needs a MODEL_PRICING entry under that same name.
        assert os.getenv(OPENROUTER_KEY_NAME), f"{OPENROUTER_KEY_NAME} environment variable is not set."
        if self.model_name not in MODEL_PRICING:  # fail before starting
            raise Exception(f"No MODEL_PRICING entry for model '{self.model_name}'.")
        self._changed: set[str] = set()

    # -- running the agent --------------------------------------------------

    def invoke(self, prompt: str) -> None:
        """Run the conversation in a separate process, and stop it at the budget.

        OpenHands runs its tools in whichever Python process runs the
        conversation, and the tools must run as the agent's user, so the
        conversation gets a process of its own (``run_conversation``).
        """
        report_r, report_w = os.pipe()
        env = self.agent_env(PYTHONPATH=os.environ.get("PYTHONPATH", ""),
                             PATCHBENCH_REPORT_FD=str(report_w),
                             **{OPENROUTER_KEY_NAME: os.environ[OPENROUTER_KEY_NAME]})
        conversation = subprocess.Popen([sys.executable, __file__, "--conversation"],
                                        stdin=subprocess.PIPE, text=True, env=env,
                                        pass_fds=(report_w,), **self.as_agent())
        os.close(report_w)
        conversation.stdin.write(json.dumps({"model_name": self.model_name,
                                             "effort": self.effort,
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
        if conversation.returncode not in (0, -signal.SIGTERM):
            print(f"Conversation process exited with code {conversation.returncode}", flush=True)

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

    Reads the model, effort, checkout and prompt as JSON on stdin. After every
    event it reports the spend so far, and each file a ``file_editor`` command
    wrote, one JSON object per line on ``PATCHBENCH_REPORT_FD``.
    """
    job = json.load(sys.stdin)
    reports = os.fdopen(int(os.environ["PATCHBENCH_REPORT_FD"]), "w", buffering=1)
    pricing = MODEL_PRICING[job["model_name"]]

    key = os.environ[OPENROUTER_KEY_NAME]
    # litellm reads the key from the environment as well as the config.
    os.environ["LLM_API_KEY"] = key
    llm_kwargs = {"model": _litellm_model(job["model_name"]),
                  "api_key": key,
                  "base_url": OPENROUTER_BASE_URL}
    if job.get("effort"):
        llm_kwargs["reasoning_effort"] = job["effort"]
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