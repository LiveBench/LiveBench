"""Harness handler for `moment/luxon` (agentic_coding_v2 / javascript).

Test runner: Jest 29 via babel-jest (no build step — babel transforms src/ on the
fly). The baked test command (config.yaml `test_command`) is
`TZ=America/New_York node ./node_modules/.bin/jest --verbose --runInBand`. The TZ
is MANDATORY: luxon's suite asserts local-zone behavior against America/New_York
(its CI pins `TZ: America/New_York`); any other zone diverges. Time-relative tests
are safe because test/helpers.js `withNow` freezes Settings.now.

Jest --verbose (non-TTY → no ANSI) prints suite-grouped output:
    PASS test/datetime/create.test.js
      DateTime
        ✓ DateTime.local() has now() as its zone (2 ms)
    FAIL test/datetime/tokenParse.test.js
        ✕ ... makes dots optional and handles non breakable spaces (4 ms)

Names are "<suite>:<test>" where <suite> is the most recent PASS/FAIL <file>
header; the suite headers themselves are ALSO recorded (a bare "<file>" entry),
matching iamkun/dayjs and parse_jest_suite() in
question_generation/agentic_coding_v2/javascript/scripts/04_validate_prs.py — this
parse_log MUST stay byte-identical to it. Fresh repo (no legacy handler) → always
builds from the pre-built per-PR image via TypeScriptCustomBuildImage.
"""

import re
from typing import Optional

from livebench.agentic_code_runner.eval.harness.image import (
    Config,
    Image,
    TypeScriptCustomBuildImage,
)
from livebench.agentic_code_runner.eval.harness.instance import Instance, TestResult
from livebench.agentic_code_runner.eval.harness.pull_request import PullRequest


@Instance.register("moment", "luxon")
class Luxon(Instance):
    def __init__(self, pr: PullRequest, config: Config, *args, **kwargs):
        super().__init__()
        self._pr = pr
        self._config = config

    @property
    def pr(self) -> PullRequest:
        return self._pr

    def dependency(self) -> Optional[Image]:
        prefix = self._pr.image_prefix or "javascript_abacus"
        return TypeScriptCustomBuildImage(self.pr, self._config, base_prefix=prefix)

    def fix_patch_run(self, fix_patch_run_cmd: str = "") -> str:
        if fix_patch_run_cmd:
            return fix_patch_run_cmd
        return "bash /home/fix-run.sh"

    def parse_log(self, test_log: str) -> TestResult:
        passed_tests: set[str] = set()
        failed_tests: set[str] = set()
        skipped_tests: set[str] = set()

        current_suite = None

        # Strip captured names: Jest emits the per-test "(N ms)" timing
        # inconsistently and some titles have trailing spaces, so an un-normalised
        # capture yields two different names for the same test across runs and
        # silently contaminates FAIL_TO_PASS. Mirrors parse_jest_suite() in
        # scripts/04_validate_prs.py (and iamkun/dayjs.py). luxon questions always
        # carry image_prefix, so names are always stripped.
        re_pass_suite = re.compile(r"^PASS (.+?)(?:\s\(\d*\.?\d+\s*\w+\))?$")
        re_pass_test = re.compile(r"^✓ (.+?)(?:\s\(\d*\.?\d+\s*\w+\))?$")
        re_fail_suite = re.compile(r"^FAIL (.+?)(?:\s\(\d*\.?\d+\s*\w+\))?$")
        re_fail_test = re.compile(r"^✕ (.+?)(?:\s\(\d*\.?\d+\s*\w+\))?$")

        for line in test_log.splitlines():
            line = line.strip()
            if not line:
                continue

            m = re_pass_suite.match(line)
            if m:
                current_suite = m.group(1).strip()
                passed_tests.add(current_suite)

            m = re_fail_suite.match(line)
            if m:
                current_suite = m.group(1).strip()
                failed_tests.add(current_suite)

            m = re_pass_test.match(line)
            if m:
                if current_suite is None:
                    continue
                passed_tests.add(f"{current_suite}:{m.group(1).strip()}")

            m = re_fail_test.match(line)
            if m:
                if current_suite is None:
                    continue
                failed_tests.add(f"{current_suite}:{m.group(1).strip()}")

        return TestResult(
            passed_count=len(passed_tests),
            failed_count=len(failed_tests),
            skipped_count=len(skipped_tests),
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
        )
