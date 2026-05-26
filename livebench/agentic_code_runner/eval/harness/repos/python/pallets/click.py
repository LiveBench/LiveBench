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
        return CustomBuildImage(self.pr, self._config)

    def fix_patch_run(self, fix_patch_run_cmd: str = "") -> str:
        if fix_patch_run_cmd:
            return fix_patch_run_cmd
        return "bash /home/fix-run.sh"

    def parse_log(self, log: str) -> TestResult:
        test_status_map = {}
        for line in log.split("\n"):
            if any(line.startswith(x.value) for x in TestStatus):
                if line.startswith(TestStatus.FAILED.value):
                    line = line.replace(" - ", " ")
                parts = line.split()
                if len(parts) <= 1:
                    continue
                test_status_map[parts[1]] = parts[0]
        return mapping_to_testresult(test_status_map)
