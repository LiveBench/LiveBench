from typing import Optional

from livebench.agentic_code_runner.eval.harness.image import Config, Image, SWEImageDefault
from livebench.agentic_code_runner.eval.harness.instance import Instance, TestResult
from livebench.agentic_code_runner.eval.harness.pull_request import PullRequest
from livebench.agentic_code_runner.eval.harness.test_result import TestStatus, mapping_to_testresult


@Instance.register("pydata", "xarray")
class Xarray(Instance):
    def __init__(self, pr: PullRequest, config: Config, *args, **kwargs):
        super().__init__()
        self._pr = pr
        self._config = config

    @property
    def pr(self) -> PullRequest:
        return self._pr

    def dependency(self) -> Optional[Image]:
        return SWEImageDefault(self.pr, self._config)

    def fix_patch_run(self, fix_patch_run_cmd: str = "") -> str:
        if fix_patch_run_cmd:
            return fix_patch_run_cmd

        return "bash /home/fix-run.sh"

    def parse_log(self, log: str) -> TestResult:
        test_status_map = {}
        for line in log.split("\n"):
            if any([line.startswith(x.value) for x in TestStatus]):
                # Additional parsing for FAILED status
                if line.startswith(TestStatus.FAILED.value):
                    line = line.replace(" - ", " ")
                test_case = line.split()
                if len(test_case) <= 1:
                    continue
                test_status_map[test_case[1]] = test_case[0]

        return mapping_to_testresult(test_status_map)
