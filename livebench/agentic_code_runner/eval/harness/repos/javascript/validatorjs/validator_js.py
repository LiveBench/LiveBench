"""Harness handler for `validatorjs/validator.js` (agentic_coding_v2 / javascript).

Test runner: Mocha (spec reporter), `mocha --require @babel/register`. Tests
import build artifacts, so base.ref runs `npm run build` first. Output:
  Validators                         describe header (plain indented text)
    ✓ should validate email          pass
    2) should reject bad input        fail (cumulative N) prefix, in-tree)
    - skipped one                     pending/skip
  249 passing (205ms)                 summary — parsing STOPS here

Names are the test TITLE; Mocha allows duplicate it() titles (~10 in
validator.js) so duplicates merge worst-status-wins. This is a fresh repo (no
legacy handler), so it always builds from the pre-built per-PR image via
TypeScriptCustomBuildImage. The parser MUST stay byte-identical to parse_mocha()
in question_generation/agentic_coding_v2/javascript/scripts/04_validate_prs.py.
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
from livebench.agentic_code_runner.eval.harness.test_result import mapping_to_testresult

_RE_M_ANSI    = re.compile(r"\x1b\[[0-9;]*[mGKHFABCDJst]")
_RE_M_PASS    = re.compile(r"^\s*✓ (.+?)(?:\s+\(\d+ms\))?$")
_RE_M_FAIL    = re.compile(r"^\s*\d+\) (.+?)$")
_RE_M_PEND    = re.compile(r"^\s*- (.+?)$")
_RE_M_SUMMARY = re.compile(r"^\s*\d+ (passing|failing|pending)\b")
_M_SEVERITY   = {"PASSED": 0, "SKIPPED": 1, "FAILED": 2}


@Instance.register("validatorjs", "validator.js")
class ValidatorJs(Instance):
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

    def parse_log(self, log: str) -> TestResult:
        results: dict[str, str] = {}

        def record(name: str, status: str) -> None:
            name = name.strip()
            if not name:
                return
            prev = results.get(name)
            if prev is None or _M_SEVERITY[status] > _M_SEVERITY[prev]:
                results[name] = status

        for raw_line in log.splitlines():
            line = _RE_M_ANSI.sub("", raw_line)
            if _RE_M_SUMMARY.match(line):
                break
            m = _RE_M_PASS.match(line)
            if m:
                record(m.group(1), "PASSED")
                continue
            m = _RE_M_FAIL.match(line)
            if m:
                record(m.group(1), "FAILED")
                continue
            m = _RE_M_PEND.match(line)
            if m:
                record(m.group(1), "SKIPPED")
        return mapping_to_testresult(results)
