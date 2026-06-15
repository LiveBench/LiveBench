"""Harness handler for `nuxt/nuxt` (agentic_coding_v2 typescript_abacus images).

Replaces the legacy mswebench handler (node:20 + fresh clone + suite-level
parser). Our questions run in prebuilt `typescript_abacus/nuxt_m_nuxt:pr-N`
images via the TypeScriptCustomBuildImage pattern; the production fix-run.sh
re-runs `pnpm dev:prepare` and vitest inside the container.

Test runner: Vitest (verbose reporter).  Output lines look like:
  " ✓  nuxt  test/nuxt/composables.test.ts > suite > test name  5ms"
  " ×  unit  packages/nuxt/test/pages.test.ts > suite > test name"
Older vitest versions render the project prefix as "|nuxt|" instead of the
badge form — _clean() normalises it.  ANSI escape codes are stripped before
matching.
"""

import fnmatch
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
# These regexes MUST stay in sync with the Vitest/Jest log parser used
# during PR validation (04_validate_prs.py parse_vitest_jest).

_RE_ANSI   = re.compile(r"\x1b\[[0-9;]*[mGKHFABCDJst]")
_RE_PASS   = re.compile(r"^\s*[✓√] (.+?)(?:\s+\d+(?:\.\d+)?(?:ms|s))?$")
_RE_FAIL   = re.compile(r"^\s*[×✕✗] (.+?)(?:\s+\d+(?:\.\d+)?(?:ms|s))?$")
_RE_SKIP   = re.compile(r"^\s*↓ (.+?)(?:\s+\[skipped\])?$")
_RE_TIMING = re.compile(r"\s+\d+(?:\.\d+)?(?:ms|s)$")
_RE_RETRY_TIMING = re.compile(r"\s+\d+(?:\.\d+)?(?:ms|s)\s+\(retry x\d+\)$")
_RE_RETRY_ONLY   = re.compile(r"\s+\(retry x\d+\)$")
_RE_JEST_TIMING  = re.compile(r"\s+\(\d+(?:\.\d+)?ms\)$")
_RE_WS     = re.compile(r"\s+")
_RE_PROJ_PIPE = re.compile(r"^\|([^|]+)\|\s*")


# Known-flaky test families (timing-dependent), excluded from parse output
# entirely. MUST stay in sync with flaky_tests in the question-generation
# repos/nuxt/config.yaml. These are already absent from every question's
# FAIL_TO_PASS/PASS_TO_PASS; filtering here keeps a random flip from
# tripping gen_report's run/test/fix consistency checks, which would
# otherwise invalidate the report and zero a correct model patch
# (observed: pr-34320, scrollBehavior flake PASS->FAIL in fix phase).
_FLAKY_GLOBS = (
    "* test/nuxt/router.options.test.ts > scrollBehavior of router options with global transition > *",
    "unit packages/nuxt/test/builder.test.ts > builder:watch > *",
)


def _is_flaky(name: str) -> bool:
    return any(fnmatch.fnmatchcase(name, g) for g in _FLAKY_GLOBS)


def _clean(name: str) -> str:
    name = _RE_RETRY_TIMING.sub("", name)
    name = _RE_RETRY_ONLY.sub("", name)
    name = _RE_JEST_TIMING.sub("", name)
    name = _RE_TIMING.sub("", name)
    name = _RE_WS.sub(" ", name).strip()
    # Normalise old-vitest "|project|" prefix to the badge form "project "
    # (must stay in sync with parse_vitest_jest in 04_validate_prs.py).
    return _RE_PROJ_PIPE.sub(r"\1 ", name)


@Instance.register("nuxt", "nuxt")
class Nuxt(Instance):
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
        for name in [n for n in test_status_map if _is_flaky(n)]:
            del test_status_map[name]
        return mapping_to_testresult(test_status_map)
