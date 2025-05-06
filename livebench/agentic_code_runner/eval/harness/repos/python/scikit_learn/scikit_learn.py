import re
from typing import Optional

from multi_swe_bench.harness.image import Config, Image, SWEImageDefault
from multi_swe_bench.harness.instance import Instance, TestResult
from multi_swe_bench.harness.pull_request import PullRequest
from multi_swe_bench.harness.test_result import TestStatus, mapping_to_testresult


@Instance.register("scikit-learn", "scikit-learn")
class ScikitLearn(Instance):
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
        escapes = "".join([chr(char) for char in range(1, 32)])
        for line in log.split("\n"):
            line = re.sub(r"\[(\d+)m", "", line)
            translator = str.maketrans("", "", escapes)
            line = line.translate(translator)
            if any([line.startswith(x.value) for x in TestStatus]):
                if line.startswith(TestStatus.FAILED.value):
                    line = line.replace(" - ", " ")
                test_case = line.split()
                if len(test_case) >= 2:
                    test_status_map[test_case[1]] = test_case[0]
            # Support older pytest versions by checking if the line ends with the test status
            elif any([line.endswith(x.value) for x in TestStatus]):
                test_case = line.split()
                if len(test_case) >= 2:
                    test_status_map[test_case[0]] = test_case[1]

        return mapping_to_testresult(test_status_map)
