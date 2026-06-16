"""Harness handler for `ajv-validator/ajv` (agentic_coding_v2 / javascript).

Test runner: Mocha 10 (spec reporter) via ts-node, `mocha -r ts-node/register
--reporter spec "spec/**/*.spec.{ts,js}"`. Specs import the freshly-built dist/,
so the baked test command (config.yaml `test_command`) first patches tsconfig
with skipLibCheck, runs `npm run build` (tsc), self-links node_modules/ajv to the
local build (a stale published copy is pulled in by a devDep), regenerates the
JSON-Schema-Test-Suite fixtures (`npm run json-tests`), then runs Mocha.

Mocha 10 spec output (non-TTY → ANSI stripped here):
    Validators                       describe header (plain indented text)
      ✔ should validate email        pass  (mocha 10 glyph is ✔ U+2714)
      2) should reject bad input      fail  (cumulative N) prefix, in-tree)
      - skipped one                   pending/skip
    7594 passing (12s)               summary — parsing STOPS here

Names are the test TITLE; ajv allows duplicate it() titles so duplicates merge
worst-status-wins. Fresh repo (no legacy handler) → always builds from the
pre-built per-PR image via TypeScriptCustomBuildImage. The parser MUST stay
byte-identical to parse_mocha() in
question_generation/agentic_coding_v2/javascript/scripts/04_validate_prs.py —
note the pass glyph is the superset [✓✔] (✓ for mocha 6 / validator.js, ✔ for
mocha 10 / ajv).
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
_RE_M_PASS    = re.compile(r"^\s*[✓✔] (.+?)(?:\s+\(\d+ms\))?$")
_RE_M_FAIL    = re.compile(r"^\s*\d+\) (.+?)$")
_RE_M_PEND    = re.compile(r"^\s*- (.+?)$")
_RE_M_SUMMARY = re.compile(r"^\s*\d+ (passing|failing|pending)\b")
_M_SEVERITY   = {"PASSED": 0, "SKIPPED": 1, "FAILED": 2}


@Instance.register("ajv-validator", "ajv")
class Ajv(Instance):
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
