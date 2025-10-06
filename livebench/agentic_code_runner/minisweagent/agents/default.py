"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation."""

import re
import subprocess
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
    action_observation_template: str = "Observation: {{output}}"
    step_limit: int = 0
    cost_limit: float = 0


class NonTerminatingException(Exception):
    """Raised for conditions that can be handled by the agent."""


class FormatError(NonTerminatingException):
    """Raised when the LM's output is not in the expected format."""


class ExecutionTimeoutError(NonTerminatingException):
    """Raised when the action execution timed out."""


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

    def render_template(self, template: str, **kwargs) -> str:
        template_vars = asdict(self.config) | self.env.get_template_vars() | self.model.get_template_vars()
        return Template(template, undefined=StrictUndefined).render(
            **kwargs, **template_vars, **self.extra_template_vars
        )

    def add_message(self, role: str, content: str, **kwargs):
        self.messages.append({"role": role, "content": content, **kwargs})

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Run step() until agent is finished. Return exit status & message"""
        out = self.env.execute("git add -A && git diff --cached")
        if out["returncode"] != 0:
            raise RuntimeError(f"Error checking for existing changes: {out}")
        self.existing_git_diff = out["output"]
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.add_message("system", self.render_template(self.config.system_template))
        self.add_message("user", self.render_template(self.config.instance_template))
        while True:
            try:
                self.step()
            except NonTerminatingException as e:
                logger.warning(f"Non-terminating exception: {e}")
                self.add_message("user", str(e))
            except TerminatingException as e:
                logger.warning(f"Terminating exception: {type(e).__name__}")
                self.add_message("user", str(e))
                return type(e).__name__, str(e)
            except Exception as e:
                logger.warning(f"Exception: {e}")
                traceback.print_exc()
                self.add_message("user", str(e))
                return "UnknownError " + type(e).__name__, str(e)

    def step(self) -> dict:
        """Query the LM, execute the action, return the observation."""
        return self.get_observation(self.query())

    def query(self) -> dict:
        """Query the model and return the response."""
        if 0 < self.config.step_limit <= self.model.n_calls:
            logger.info(f"Autosubmitting after Limits exceeded: {self.model.n_calls} steps")
            out = self.env.execute("git add -A && git diff --cached")
            if out["returncode"] != 0:
                raise RuntimeError(f"Error checking for existing changes: {out}")
            new_diff = out["output"]
            if self.existing_git_diff != "":
                # need to remove the changes from the existing diff from the new diff
                # so that the final diff only includes the changes from the agent
                new_diff = extract_new_changes(self.existing_git_diff, new_diff)
            raise LimitsExceeded(new_diff)
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
        actions = re.findall(r"```bash\s*\n(.*?)\n```", response["content"], re.DOTALL)
        if len(actions) == 1:
            return {"action": actions[0].strip(), **response}
        raise FormatError(self.render_template(self.config.format_error_template, actions=actions))

    def execute_action(self, action: dict) -> dict:
        try:
            output = self.env.execute(action["action"])
        except subprocess.TimeoutExpired as e:
            output = e.output.decode("utf-8", errors="replace") if e.output else ""
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
            new_diff = "".join(lines[1:])
            if self.existing_git_diff != "":
                # need to remove the changes from the existing diff from the new diff
                # so that the final diff only includes the changes from the agent
                new_diff = extract_new_changes(self.existing_git_diff, new_diff)
            raise Submitted(new_diff)
