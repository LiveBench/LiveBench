"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation."""

import os
import re
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
import traceback

from jinja2 import StrictUndefined, Template

from livebench.agentic_code_runner.minisweagent import Environment, Model
from livebench.agentic_code_runner.minisweagent.utils.log import logger


@dataclass
class AgentConfig:
    # The default settings are the bare minimum to run the agent. Take a look at the config files for improved settings.
    system_template: str = "You are a helpful assistant that can do anything."
    instance_template: str = (
        "Your task: {{task}}. Please reply with a single shell command in triple backticks. "
        "To finish, the first line of the output of the shell command must be 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'."
    )
    timeout_template: str = (
        "The last command <command>{{action['action']}}</command> timed out and has been killed.\n"
        "The output of the command was:\n <output>\n{{output}}\n</output>\n"
        "Please try another command and make sure to avoid those requiring interactive input."
    )
    format_error_template: str = "Please always provide EXACTLY ONE action in triple backticks."
    noop_action_template: str = (
        "Your command was a no-op (`true`, `:`, or empty) and did nothing — no progress was made. "
        "If your fix is already complete, submit now by making the FIRST line of a command's output "
        "'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT' (e.g. `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git diff`). "
        "Otherwise, run a real command that makes progress on the task."
    )
    action_observation_template: str = "Observation: {{output}}"
    empty_submission_template: str = (
        "Your submission produces an EMPTY diff against the base commit: you have NOT actually "
        "modified any source files, so the task is not solved.\n"
        "Do NOT assume the environment is unreliable, stale, lagged, or intermittent — command "
        "output is reliable. Do NOT claim a fix is already staged or that tests passed unless you "
        "have just seen that exact output.\n"
        "Recover step by step, one shell command per turn:\n"
        "1. Run `git diff` to confirm there are currently no changes.\n"
        "2. Open the relevant source file(s) and make the actual code edit that fixes the issue.\n"
        "3. Run the failing test(s) and read the output to verify the fix works.\n"
        "4. Only then resubmit with COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT."
    )
    step_limit: int = 0
    cost_limit: float = 0
    time_limit: float = 0
    """Wall-clock budget per instance in seconds (0 = disabled). Exceeding it
    raises LimitsExceeded, so the current diff is auto-submitted exactly like
    hitting the step limit."""


class NonTerminatingException(Exception):
    """Raised for conditions that can be handled by the agent."""


class FormatError(NonTerminatingException):
    """Raised when the LM's output is not in the expected format."""


class NoOpActionError(NonTerminatingException):
    """Raised when the LM emits a do-nothing command (`true`, `:`, or empty).

    Handled like other NonTerminatingExceptions: the message is appended as a user
    turn nudging the model to do real work or submit. Because it is raised after
    model.query() (n_calls already incremented), each nudge counts as a step, so the
    loop stays bounded by step_limit exactly like a format error."""


class EmptySubmissionError(NonTerminatingException):
    """Raised when the LM tries to submit but its diff vs the base commit is empty.

    Handled like other NonTerminatingExceptions: the message is appended as a user
    turn nudging the model to make a real change, and the loop continues (bounded by
    MSWEA_EMPTY_SUBMISSION_RETRIES and the step limit)."""


class ExecutionTimeoutError(NonTerminatingException):
    """Raised when the action execution timed out."""


_TIMEOUT_OUTPUT_CAP = 10_000


def _truncate_middle(text: str, cap: int = _TIMEOUT_OUTPUT_CAP) -> str:
    """Bound a runaway command's captured output, keeping head and tail."""
    if len(text) <= cap:
        return text
    half = cap // 2
    return f"{text[:half]}\n... [{len(text) - cap:,} chars truncated] ...\n{text[-half:]}"


class TerminatingException(Exception):
    """Raised for conditions that terminate the agent."""


class Submitted(TerminatingException):
    """Raised when the LM declares that the agent has finished its task."""


class LimitsExceeded(TerminatingException):
    """Raised when the agent has reached its cost or step limit."""

def extract_new_changes(old_diff: str, new_diff: str) -> str:
    """Remove pre-existing changes from new_diff, keeping only new changes."""
    
    # Split into file sections (each starts with "diff --git")
    def parse_diff(diff_text: str) -> dict[str, str]:
        files: dict[str, str] = {}
        current_file = None
        current_content = []
        
        for line in diff_text.split('\n'):
            if line.startswith('diff --git'):
                if current_file:
                    files[current_file] = '\n'.join(current_content)
                current_file = line.rstrip()
                current_content = [line.rstrip()]
            elif current_file:
                current_content.append(line.rstrip())
        
        if current_file:
            files[current_file] = '\n'.join(current_content)
        
        return files
    
    old_files = parse_diff(old_diff)
    new_files = parse_diff(new_diff)
    # Keep only files/changes that differ from old diff
    result: list[str] = []
    for file_header, content in new_files.items():
        if file_header not in old_files or old_files[file_header].strip() != content.strip():
            result.append(content)
    
    return '\n'.join(result)

class DefaultAgent:
    def __init__(self, model: Model, env: Environment, *, config_class: Callable = AgentConfig, **kwargs):
        self.config: AgentConfig = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars = {}
        self.existing_git_diff = ""
        self.base_commit = ""
        # Empty-submission guard (see has_finished). Tuned by MSWEA_EMPTY_SUBMISSION_RETRIES
        # (default 3 -> guard ON so empty submissions never silently slip through a run;
        # set to 0 to disable). Hard-capped at 3 so a misconfig can't loop the step budget away.
        self.max_empty_submission_retries = min(int(os.getenv("MSWEA_EMPTY_SUBMISSION_RETRIES", "3")), 3)
        self.empty_submissions = 0  # confirmed-empty submissions that were nudged this run
        self.empty_submissions_recovered = 0  # runs where a nudge later yielded a real diff
        self._empty_submission_nudges = 0  # per-run counter, reset in run()

    def render_template(self, template: str, **kwargs) -> str:
        template_vars = asdict(self.config) | self.env.get_template_vars() | self.model.get_template_vars()
        return Template(template, undefined=StrictUndefined).render(
            **kwargs, **template_vars, **self.extra_template_vars
        )

    def add_message(self, role: str, content: str, **kwargs):
        self.messages.append({"role": role, "content": content, **kwargs})

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Run step() until agent is finished. Return exit status & message"""
        out = self.env.execute("git rev-parse HEAD")
        if out["returncode"] == 0:
            self.base_commit = out["output"].strip().splitlines()[-1].strip()
        else:
            logger.warning(f"Could not record base commit, falling back to HEAD-relative diffs: {out}")
            self.base_commit = ""
        out = self.env.execute(self._diff_command())
        if out["returncode"] != 0:
            raise RuntimeError(f"Error checking for existing changes: {out}")
        self.existing_git_diff = out["output"]
        self.extra_template_vars |= {"task": task, **kwargs}
        self._start_time = time.monotonic()
        self.messages = []
        self._empty_submission_nudges = 0
        self.add_message("system", self.render_template(self.config.system_template))
        self.add_message("user", self.render_template(self.config.instance_template))
        while True:
            try:
                self.step()
            except NonTerminatingException as e:
                # Cap the logged copy: rendering a multi-MB message through
                # rich's highlighter livelocks the whole batch (it holds the
                # global logging lock). The full text still reaches the model.
                logger.warning(f"Non-terminating exception: {str(e)[:2000]}")
                self.add_message("user", str(e))
            except Submitted as e:
                # has_finished() already captured the submission diff
                logger.warning(f"Terminating exception: {type(e).__name__}")
                self.add_message("user", str(e))
                return type(e).__name__, str(e)
            except TerminatingException as e:
                # For other terminating exceptions (LimitsExceeded), capture git diff as submission
                logger.warning(f"Terminating exception: {type(e).__name__}")
                diff = self._capture_git_diff() or ""
                self.add_message("user", str(e))
                return type(e).__name__, diff
            except RuntimeError as e:
                # Check if this is a global cost/call limit - if so, terminate with git diff
                if "Global cost/call limit exceeded" in str(e):
                    logger.warning(f"Limit exception (terminating): {type(e).__name__}: {e}")
                    diff = self._capture_git_diff() or ""
                    self.add_message("user", str(e))
                    return type(e).__name__, diff
                # Other RuntimeErrors should propagate
                raise
            except Exception as e:
                # All other exceptions terminate the agent and capture git diff
                logger.warning(f"Unhandled exception (terminating): {type(e).__name__}: {e}")
                traceback.print_exc()
                diff = self._capture_git_diff() or ""
                self.add_message("user", str(e))
                return type(e).__name__, diff
    
    def _diff_command(self) -> str:
        # Diff against the base commit recorded at run start, not just the index vs
        # HEAD: if the agent committed during the trajectory, `git diff --cached`
        # alone is empty and the work would be lost.
        if self.base_commit:
            return f"git add -A && git diff --cached {self.base_commit}"
        return "git add -A && git diff --cached"

    def _capture_git_diff(self) -> str | None:
        """Capture current git diff vs the base commit, removing pre-existing changes.

        Returns None if the capture itself failed (as opposed to a genuinely empty diff).
        """
        try:
            out = self.env.execute(self._diff_command())
            if out["returncode"] == 0:
                new_diff = out["output"]
                if self.existing_git_diff != "":
                    new_diff = extract_new_changes(self.existing_git_diff, new_diff)
                return new_diff
            logger.warning(f"Failed to capture git diff: {out}")
        except Exception as e:
            logger.warning(f"Failed to capture git diff: {e}")
        return None

    def step(self) -> dict:
        """Query the LM, execute the action, return the observation."""
        return self.get_observation(self.query())

    def query(self) -> dict:
        """Query the model and return the response.
        
        If step limit is exceeded, we capture the current git diff and raise LimitsExceeded
        with the diff as the result.
        """
        if 0 < self.config.step_limit <= self.model.n_calls:
            logger.info(f"Autosubmitting after step limit exceeded: {self.model.n_calls} steps")
            raise LimitsExceeded("Step limit exceeded")
        elapsed = time.monotonic() - getattr(self, "_start_time", time.monotonic())
        if 0 < self.config.time_limit <= elapsed:
            logger.info(f"Autosubmitting after time limit exceeded: {elapsed:.0f}s over {self.model.n_calls} steps")
            raise LimitsExceeded("Time limit exceeded")
        response = self.model.query(self.messages)
        self.add_message("assistant", **response)
        return response

    def get_observation(self, response: dict) -> dict:
        """Execute the action and return the observation."""
        output = self.execute_action(self.parse_action(response))
        observation = self.render_template(self.config.action_observation_template, output=output)
        self.add_message("user", observation)
        return output

    def parse_action(self, response: dict) -> dict:
        """Parse the action from the message. Returns the action."""
        content = response["content"]
        if '<think>' in content and '</think>' in content:
            content = content.split('</think>')[1].strip() # don't parse actions from thinking content
        actions = re.findall(r"```bash\s*\n(.*?)\n```", content, re.DOTALL)
        if len(actions) == 0:
            actions = re.findall(r"```bash\s*\n(.*?)```", content, re.DOTALL)
        if len(actions) >= 1:
            # Tolerate models that bundle several bash blocks (or simulate the whole
            # loop) in one turn: execute the FIRST block and feed the real observation
            # back, rather than rejecting the entire turn. The model then continues from
            # real output instead of its own hallucinated output.
            action = actions[0].strip()
            # No-op guard (always on): a bare `true`/`:`/empty command accomplishes nothing
            # and is the hallmark of the idle loop where a model believes it already
            # submitted and spins to the step limit. Don't execute it; feed back an explicit
            # nudge instead. Exact-match only, so real commands (e.g. `x || true`) are
            # untouched, and none of these can contain the submit sentinel. This is raised
            # after model.query() ran, so it counts as a step -> bounded by step_limit and
            # can never make a run worse (same terminal state, more useful feedback).
            if action in ("true", ":", ""):
                raise NoOpActionError(self.render_template(self.config.noop_action_template))
            return {"action": action, **response}
        raise FormatError(self.render_template(self.config.format_error_template, actions=actions))

    def execute_action(self, action: dict) -> dict:
        try:
            output = self.env.execute(action["action"])
        except subprocess.TimeoutExpired as e:
            output = e.output.decode("utf-8", errors="replace") if e.output else ""
            # A command that floods stdout until the timeout (e.g. an agent
            # reproducing an infinite-loop bug) can produce hundreds of MB
            # here; unbounded, it ends up in the exception message and from
            # there in both the log record and the model context.
            output = _truncate_middle(output)
            raise ExecutionTimeoutError(
                self.render_template(self.config.timeout_template, action=action, output=output)
            )
        except TimeoutError:
            raise ExecutionTimeoutError(self.render_template(self.config.timeout_template, action=action, output=""))
        self.has_finished(output)
        return output

    def has_finished(self, output: dict[str, str]):
        """Raises Submitted exception with final output if the agent has finished its task."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() in ["MINI_SWE_AGENT_FINAL_OUTPUT", "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]:
            # Recapture the diff from the environment rather than trusting the
            # model-printed output: the printed `git diff --cached` is empty if the
            # agent committed during the trajectory, and may be truncated or echoed.
            captured = self._capture_git_diff()
            if captured is None:
                # capture failed; fall back to the model-printed diff. We can't prove
                # the tree is empty on this path, so the empty-submission guard below
                # is intentionally skipped.
                new_diff = "".join(lines[1:])
                if self.existing_git_diff != "":
                    # need to remove the changes from the existing diff from the new diff
                    # so that the final diff only includes the changes from the agent
                    new_diff = extract_new_changes(self.existing_git_diff, new_diff)
                raise Submitted(new_diff)

            new_diff = captured
            # Empty-submission guard: a confirmed-empty diff means the agent is
            # submitting without having changed any source file, often after
            # hallucinating that a fix was already staged or that the environment was
            # returning stale/lagged output. Rather than accept the empty submission,
            # nudge the model to make + verify a real edit and continue the loop.
            # Bounded by max_empty_submission_retries; on exhaustion we accept the
            # empty diff so the worst case is unchanged.
            if new_diff.strip() == "":
                if self._empty_submission_nudges < self.max_empty_submission_retries:
                    self.empty_submissions += 1
                    self._empty_submission_nudges += 1
                    logger.warning(
                        "Empty submission (no diff vs base commit); nudging the model to make a real "
                        f"change (attempt {self._empty_submission_nudges}/{self.max_empty_submission_retries})"
                    )
                    raise EmptySubmissionError(self.render_template(self.config.empty_submission_template))
            elif self._empty_submission_nudges > 0:
                # An earlier submission this run was empty; the nudge yielded a real diff.
                self.empty_submissions_recovered += 1
            raise Submitted(new_diff)
