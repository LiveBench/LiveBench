"""Harness handler for `hgrecco/pint` (agentic_coding_v2/python).

pint is pure-Python units/quantities; tests are pytest under `pint/testsuite/`.
The pytest `parse_log` is identical to the other python_abacus handlers
(attrs/marshmallow/...) — no new parser needed. `dependency()` reads
`image_prefix` from the question row so the same handler works for any prefix
(falls back to "python_abacus", where pint's per-PR images live).
"""

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


@Instance.register("hgrecco", "pint")
class Pint(Instance):
    def __init__(self, pr: PullRequest, config: Config, *args, **kwargs):
        super().__init__()
        self._pr = pr
        self._config = config

    @property
    def pr(self) -> PullRequest:
        return self._pr

    def dependency(self) -> Optional[Image]:
        prefix = self._pr.image_prefix or "python_abacus"
        return CustomBuildImage(self.pr, self._config, base_prefix=prefix)

    def fix_patch_run(self, fix_patch_run_cmd: str = "") -> str:
        if fix_patch_run_cmd:
            return fix_patch_run_cmd
        return "bash /home/fix-run.sh"

    def parse_log(self, log: str) -> TestResult:
        """Parse pytest -rA -v --no-header output into a TestResult.

        Handles two formats emitted by pytest:
          Format A (verbose): "pint/testsuite/foo.py::test_bar PASSED [ 69%]"
          Format B (short summary): "FAILED pint/testsuite/foo.py::test_bar - Reason"
        """
        test_status_map = {}
        status_values = {x.value for x in TestStatus}

        for line in log.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Format B: "FAILED pint/testsuite/foo.py::test_bar - AssertionError: ..."
            if any(line.startswith(sv) for sv in status_values):
                if line.startswith(TestStatus.FAILED.value):
                    line = line.replace(" - ", " ")
                parts = line.split()
                if len(parts) >= 2:
                    test_status_map[parts[1]] = parts[0]

            # Format A: "pint/testsuite/foo.py::test_bar PASSED [ 69%]"
            elif "::" in line:
                parts = line.split()
                if len(parts) >= 2 and parts[1] in status_values:
                    test_status_map[parts[0]] = parts[1]

        return mapping_to_testresult(test_status_map)
