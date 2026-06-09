"""Harness handler for `pallets/click`."""

from typing import Optional

from livebench.agentic_code_runner.eval.harness.image import (
    Config,
    CustomBuildImage,
    Image,
)
from livebench.agentic_code_runner.eval.harness.instance import Instance, TestResult
from livebench.agentic_code_runner.eval.harness.pull_request import PullRequest
from livebench.agentic_code_runner.eval.harness.test_result import (
    TestStatus,
    mapping_to_testresult,
)


@Instance.register("pallets", "click")
class Click(Instance):
    def __init__(self, pr: PullRequest, config: Config, *args, **kwargs):
        super().__init__()
        self._pr = pr
        self._config = config

    @property
    def pr(self) -> PullRequest:
        return self._pr

    def dependency(self) -> Optional[Image]:
        prefix = self._pr.image_prefix or "python_v4"
        return CustomBuildImage(self.pr, self._config, base_prefix=prefix)

    def fix_patch_run(self, fix_patch_run_cmd: str = "") -> str:
        if fix_patch_run_cmd:
            return fix_patch_run_cmd
        return "bash /home/fix-run.sh"

    def parse_log(self, log: str) -> TestResult:
        test_status_map = {}
        status_values = {x.value for x in TestStatus}

        for line in log.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Format B (pytest short summary): "<STATUS> <test_path> [- reason]"
            # e.g. "FAILED tests/test_utils.py::test_foo - ValueError: ..."
            if any(line.startswith(sv) for sv in status_values):
                if line.startswith(TestStatus.FAILED.value):
                    line = line.replace(" - ", " ")
                parts = line.split()
                if len(parts) >= 2:
                    test_status_map[parts[1]] = parts[0]

            # Format A (pytest verbose): "<test_path> <STATUS> [ pct%]"
            # e.g. "tests/test_shell_completion.py::test_foo PASSED [ 69%]"
            # pytest 9.0.2+ no longer emits PASSED in the short summary, so this
            # branch is required to capture passing tests from verbose output.
            elif "::" in line:
                parts = line.split()
                if len(parts) >= 2 and parts[1] in status_values:
                    test_status_map[parts[0]] = parts[1]

        return mapping_to_testresult(test_status_map)
