import re
from typing import Optional

from multi_swe_bench.harness.image import Config, Image, SWEImageDefault
from multi_swe_bench.harness.instance import Instance, TestResult
from multi_swe_bench.harness.pull_request import PullRequest
from multi_swe_bench.harness.test_result import TestStatus, mapping_to_testresult


@Instance.register("django", "django")
class Django(Instance):
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
        lines = log.split("\n")

        prev_test = None
        for line in lines:
            line = line.strip()

            # This isn't ideal but the test output spans multiple lines
            if "--version is equivalent to version" in line:
                test_status_map["--version is equivalent to version"] = (
                    TestStatus.PASSED.value
                )

            # Log it in case of error
            if " ... " in line:
                prev_test = line.split(" ... ")[0]

            pass_suffixes = (" ... ok", " ... OK", " ...  OK")
            for suffix in pass_suffixes:
                if line.endswith(suffix):
                    # TODO: Temporary, exclusive fix for django__django-7188
                    # The proper fix should involve somehow getting the test results to
                    # print on a separate line, rather than the same line
                    if line.strip().startswith(
                        "Applying sites.0002_alter_domain_unique...test_no_migrations"
                    ):
                        line = line.split("...", 1)[-1].strip()
                    test = line.rsplit(suffix, 1)[0]
                    test_status_map[test] = TestStatus.PASSED.value
                    break
            if " ... skipped" in line:
                test = line.split(" ... skipped")[0]
                test_status_map[test] = TestStatus.SKIPPED.value
            if line.endswith(" ... FAIL"):
                test = line.split(" ... FAIL")[0]
                test_status_map[test] = TestStatus.FAILED.value
            if line.startswith("FAIL:"):
                test = line.split()[1].strip()
                test_status_map[test] = TestStatus.FAILED.value
            if line.endswith(" ... ERROR"):
                test = line.split(" ... ERROR")[0]
                test_status_map[test] = TestStatus.ERROR.value
            if line.startswith("ERROR:"):
                test = line.split()[1].strip()
                test_status_map[test] = TestStatus.ERROR.value

            if line.lstrip().startswith("ok") and prev_test is not None:
                # It means the test passed, but there's some additional output (including new lines)
                # between "..." and "ok" message
                test = prev_test
                test_status_map[test] = TestStatus.PASSED.value

        # TODO: This is very brittle, we should do better
        # There's a bug in the django logger, such that sometimes a test output near the end gets
        # interrupted by a particular long multiline print statement.
        # We have observed this in one of 3 forms:
        # - "{test_name} ... Testing against Django installed in {*} silenced.\nok"
        # - "{test_name} ... Internal Server Error: \/(.*)\/\nok"
        # - "{test_name} ... System check identified no issues (0 silenced).\nok"
        patterns = [
            r"^(.*?)\s\.\.\.\sTesting\ against\ Django\ installed\ in\ ((?s:.*?))\ silenced\)\.\nok$",
            r"^(.*?)\s\.\.\.\sInternal\ Server\ Error:\ \/(.*)\/\nok$",
            r"^(.*?)\s\.\.\.\sSystem check identified no issues \(0 silenced\)\nok$",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, log, re.MULTILINE):
                test_name = match.group(1)
                test_status_map[test_name] = TestStatus.PASSED.value

        return mapping_to_testresult(test_status_map)
