"""Harness handler for `trpc/trpc` (typescript_abacus).

Images are tagged typescript_abacus/trpc_m_trpc:pr-N, built by
typescript_abacus/scripts/04_validate_prs.py.

Test runner: Vitest (verbose reporter) run from workspace root.
Root vitest.config.ts uses `projects: ['./packages/*']` so all packages run;
tests live primarily in packages/tests/.  Output lines look like:
  " ✓  packages/tests/src/foo.test.ts > suite > test name  5ms"
  " ×  packages/tests/src/foo.test.ts > suite > test name"
ANSI escape codes are stripped before matching.
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

# ── Vitest verbose output parser ──────────────────────────────────────────
# These regexes MUST stay in sync with parse_vitest_jest() in
# typescript_abacus/scripts/04_validate_prs.py.

_RE_ANSI   = re.compile(r"\x1b\[[0-9;]*[mGKHFABCDJst]")
_RE_PASS   = re.compile(r"^\s*[✓√] (.+?)(?:\s+\d+(?:\.\d+)?(?:ms|s))?$")
_RE_FAIL   = re.compile(r"^\s*[×✕✗] (.+?)(?:\s+\d+(?:\.\d+)?(?:ms|s))?$")
_RE_SKIP   = re.compile(r"^\s*↓ (.+?)(?:\s+\[skipped\])?$")
_RE_TIMING = re.compile(r"\s+\d+(?:\.\d+)?(?:ms|s)$")
_RE_RETRY_TIMING = re.compile(r"\s+\d+(?:\.\d+)?(?:ms|s)\s+\(retry x\d+\)$")
_RE_RETRY_ONLY = re.compile(r"\s+\(retry x\d+\)$")
_RE_JEST_TIMING = re.compile(r"\s+\(\d+(?:\.\d+)?ms\)$")
_RE_WS     = re.compile(r"\s+")


def _clean(name: str) -> str:
    name = _RE_RETRY_TIMING.sub("", name)
    name = _RE_RETRY_ONLY.sub("", name)
    name = _RE_JEST_TIMING.sub("", name)
    name = _RE_TIMING.sub("", name)
    return _RE_WS.sub(" ", name).strip()


@Instance.register("trpc", "trpc")
class Trpc(Instance):
    def __init__(self, pr: PullRequest, config: Config, *args, **kwargs):
        super().__init__()
        self._pr = pr
        self._config = config

    @property
    def pr(self) -> PullRequest:
        return self._pr

    def dependency(self) -> Optional[Image]:
        return TypeScriptCustomBuildImage(self.pr, self._config, base_prefix="typescript_abacus")

    def fix_patch_run(self, fix_patch_run_cmd: str = "") -> str:
        if fix_patch_run_cmd:
            return fix_patch_run_cmd
        return "bash /home/fix-run.sh"

    def parse_log(self, log: str) -> TestResult:
        test_status_map: dict[str, str] = {}
        for raw_line in log.splitlines():
            line = _RE_ANSI.sub("", raw_line)
            m = _RE_PASS.match(line)
            if m:
                test_status_map[_clean(m.group(1))] = "PASSED"
                continue
            m = _RE_FAIL.match(line)
            if m:
                test_status_map[_clean(m.group(1))] = "FAILED"
                continue
            m = _RE_SKIP.match(line)
            if m:
                test_status_map[_clean(m.group(1))] = "SKIPPED"
        return mapping_to_testresult(test_status_map)
