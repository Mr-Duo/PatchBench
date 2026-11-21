"""Runs mini-swe-agent, via the OpenRouter API."""

import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from agent_base import AGENT_HOME, AgentHarness, main
from constants import MODEL_PRICING

#: OpenRouter's OpenAI-compatible endpoint.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
#: Env var holding the OpenRouter key.
OPENROUTER_KEY_NAME = "OPENROUTER_API_KEY"

#: Where the agent run saves its messages; copied to the log directory afterwards.
TRAJECTORY = Path(AGENT_HOME, ".mini-swe-agent", "trajectory.json")
#: Seconds one shell command may run (mini-swe-agent defaults to 30, too short for builds).
COMMAND_TIMEOUT = 600
#: How long the agent process gets to exit when stopped at the budget.
STOP_GRACE_SECONDS = 10


class MiniSweAgentHarness(AgentHarness):
    name = "minisweagent"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.model_name must be an OpenRouter slug, e.g. "anthropic/claude-sonnet-4.5",
        # with a MODEL_PRICING entry under that same name.
        assert os.getenv(OPENROUTER_KEY_NAME), f"{OPENROUTER_KEY_NAME} environment variable is not set."
        if self.model_name not in MODEL_PRICING:  # fail before starting
            raise Exception(f"No MODEL_PRICING entry for model '{self.model_name}'.")
        # A previous run's trajectory would be read as this run's.
        TRAJECTORY.unlink(missing_ok=True)

    # -- running the agent --------------------------------------------------

    def invoke(self, prompt: str) -> None:
        """Run the agent in a separate process, and stop it at the budget.

        mini-swe-agent runs its shell commands from whichever Python process
        runs it, and they must run as the agent's user, so the agent gets a
        process of its own (``run_agent``).
        """
        report_r, report_w = os.pipe()
        env = self.agent_env(
            PYTHONPATH=os.environ.get("PYTHONPATH", ""),
            PATCHBENCH_REPORT_FD=str(report_w),
            # Cost is tracked here from MODEL_PRICING; don't let litellm's own
            # (missing) price for a model abort the run.
            MSWEA_COST_TRACKING="ignore_errors",
            MSWEA_SILENT_STARTUP="1",
            # Don't fetch litellm's price table from GitHub at import.
            LITELLM_LOCAL_MODEL_COST_MAP="True",
            **{OPENROUTER_KEY_NAME: os.environ[OPENROUTER_KEY_NAME]},
        )
        agent = subprocess.Popen([sys.executable, __file__, "--agent"],
                                 stdin=subprocess.PIPE, text=True, env=env,
                                 pass_fds=(report_w,), **self.as_agent())
        os.close(report_w)
        agent.stdin.write(json.dumps({"model_name": self.model_name, "effort": self.effort,
                                      "repo_folder": self.repo_folder, "prompt": prompt}))
        agent.stdin.close()

        spend = None
        try:
            with os.fdopen(report_r) as reports:
                for line in reports:
                    report = json.loads(line)
                    if report["spend"] != spend:
                        spend = report["spend"]
                        if self.note_spend(spend):
                            break
        finally:
            if agent.poll() is None:
                agent.send_signal(signal.SIGTERM)
            try:
                agent.wait(timeout=STOP_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                agent.kill()
                agent.wait()
            if TRAJECTORY.exists():
                shutil.copy(TRAJECTORY, self.log_dir / "trajectory.json")
        if agent.returncode not in (0, -signal.SIGTERM):
            print(f"Agent process exited with code {agent.returncode}", flush=True)

    # -- changed files ------------------------------------------------------

    def changed_files(self) -> list[str]:
        """mini-swe-agent only has a shell, so no tool reports the files it wrote.

        Ask git which C/C++ files differ from the baseline instead.
        """
        return self.status_changed_files() if self.repo is not None else []


def run_agent() -> None:
    """The agent itself, run as the agent's user (see ``MiniSweAgentHarness.invoke``).

    Reads the model, effort, checkout and prompt as JSON on stdin. After every
    model call it reports the spend so far, one JSON object per line on
    ``PATCHBENCH_REPORT_FD``.
    """
    # Imported here so the harness process itself doesn't need them.
    import litellm
    import yaml
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.config import builtin_config_dir
    from minisweagent.environments.local import LocalEnvironment
    from minisweagent.models.litellm_model import LitellmModel

    job = json.load(sys.stdin)
    reports = os.fdopen(int(os.environ["PATCHBENCH_REPORT_FD"]), "w", buffering=1)
    pricing = MODEL_PRICING[job["model_name"]]
    key = os.environ[OPENROUTER_KEY_NAME]

    # -- spend: token usage of every completion, priced from MODEL_PRICING ---
    tokens = {"prompt": 0, "cache_read": 0, "cache_write": 0, "completion": 0}

    def spend() -> float:
        uncached = max(tokens["prompt"] - tokens["cache_read"], 0)
        return (uncached * pricing["input"]
                + tokens["cache_read"] * pricing["cached_input"]
                + tokens["cache_write"] * pricing.get("cache_write", 0)
                + tokens["completion"] * pricing["output"]) / 1_000_000

    real_completion = litellm.completion

    def tracked_completion(*args, **kwargs):
        response = real_completion(*args, **kwargs)
        usage = getattr(response, "usage", None)
        if usage is not None:
            details = getattr(usage, "prompt_tokens_details", None)
            tokens["prompt"] += getattr(usage, "prompt_tokens", 0) or 0
            tokens["completion"] += getattr(usage, "completion_tokens", 0) or 0
            tokens["cache_read"] += getattr(details, "cached_tokens", 0) or 0
            tokens["cache_write"] += (getattr(usage, "cache_creation_input_tokens", 0)
                                      or getattr(details, "cache_write_tokens", 0) or 0)
        reports.write(json.dumps({"spend": spend()}) + "\n")
        return response

    litellm.completion = tracked_completion  # mini-swe-agent looks it up at call time

    # -- config: mini-swe-agent's default prompts, our model and environment --
    config = yaml.safe_load((Path(builtin_config_dir) / "default.yaml").read_text()) or {}
    # Its own cost limit is off: the harness stops the run at BUDGET_LIMIT.
    agent_cfg = {**(config.get("agent") or {}), "cost_limit": 0}
    model_kwargs = {**((config.get("model") or {}).get("model_kwargs") or {}),
                    "api_key": key, "api_base": OPENROUTER_BASE_URL,
                    "drop_params": True}  # models without reasoning_effort support ignore it
    if job.get("effort"):
        model_kwargs["reasoning_effort"] = job["effort"]
    env_cfg = {**(config.get("environment") or {}),
               "cwd": job["repo_folder"], "timeout": COMMAND_TIMEOUT}

    model_name = job["model_name"]
    if not model_name.startswith("openrouter/"):
        model_name = f"openrouter/{model_name}"

    class TrajectoryAgent(DefaultAgent):
        """Saves the messages after every step, so a stopped run still leaves them."""

        def step(self, *args, **kwargs):
            try:
                return super().step(*args, **kwargs)
            finally:
                dump()

    agent = None

    def dump(outcome=None) -> None:
        TRAJECTORY.parent.mkdir(parents=True, exist_ok=True)
        TRAJECTORY.write_text(json.dumps({"outcome": outcome, "messages": agent.messages},
                                         indent=2, default=str))

    agent = TrajectoryAgent(LitellmModel(model_name=model_name, model_kwargs=model_kwargs),
                            LocalEnvironment(**env_cfg), **agent_cfg)

    # Turn the budget stop (SIGTERM) into a normal exit so the trajectory is saved.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    outcome = None
    try:
        outcome = agent.run(job["prompt"])
        print(outcome, flush=True)
    finally:
        dump(outcome)


if __name__ == "__main__":
    if sys.argv[1:] == ["--agent"]:
        run_agent()
    else:
        main(MiniSweAgentHarness)