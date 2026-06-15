"""Harness handler for `withastro/astro` (agentic_coding_v2 typescript_abacus images).

Replaces the legacy mswebench handler (node:22 + fresh clone + suite-level
parser). Our questions run in prebuilt `typescript_abacus/withastro_m_astro:pr-N`
images via the TypeScriptCustomBuildImage pattern; the production fix-run.sh
re-runs `pnpm run build:ci` and the packages/astro unit suite inside the
container.

Test runner: Node's built-in node:test with the spec reporter (astro's
`astro-scripts test` wrapper pipes `node:test/reporters` spec to stdout —
same format as markedjs/marked). Output:
  "▶ config.server"                        suite open
  "  ✔ can be passed via --config (3.7ms)" test pass
  "  ✖ some failing test (1.2ms)"          test fail
  "  ﹣ skipped one (0.1ms) # SKIP"         test skip
  "✔ config.server (58.2ms)"               suite close — same glyph as a
                                           passing test; disambiguated by
                                           matching the open-suite stack.
Test names are the full suite path joined with " > ". All ~209 unit files
run in a single process (astro-scripts bundles them into one import file),
so same-named top-level suites from different files merge worst-status-wins
(order-independent). Parsing stops at the "ℹ tests" summary because node
re-prints failing leaf lines below it.

The regexes and the parsing logic MUST stay in sync with
parse_vitest_jest() in question_generation/agentic_coding_v2/typescript/
scripts/04_validate_prs.py.
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

_RE_ANSI   = re.compile(r"\x1b\[[0-9;]*[mGKHFABCDJst]")
_RE_TIMING = re.compile(r"\s+\d+(?:\.\d+)?(?:ms|s)$")
_RE_RETRY_TIMING = re.compile(r"\s+\d+(?:\.\d+)?(?:ms|s)\s+\(retry x\d+\)$")
_RE_RETRY_ONLY   = re.compile(r"\s+\(retry x\d+\)$")
_RE_JEST_TIMING  = re.compile(r"\s+\(\d+(?:\.\d+)?ms\)$")
_RE_WS     = re.compile(r"\s+")

_RE_NT_SUITE   = re.compile(r"^(\s*)▶ (.+?)(?:\s+\(\d+(?:\.\d+)?ms\))?$")
_RE_NT_PASS    = re.compile(r"^(\s*)✔ (.+?)(?:\s+\(\d+(?:\.\d+)?ms\))?(?:\s+# .*)?$")
_RE_NT_FAIL    = re.compile(r"^(\s*)✖ (.+?)(?:\s+\(\d+(?:\.\d+)?ms\))?(?:\s+# .*)?$")
_RE_NT_SKIP    = re.compile(r"^(\s*)﹣ (.+?)(?:\s+\(\d+(?:\.\d+)?ms\))?(?:\s+# .*)?$")
_RE_NT_SUMMARY = re.compile(r"^ℹ tests \d+")

_NT_SEVERITY = {"PASSED": 0, "SKIPPED": 1, "FAILED": 2}

# Known-flaky test families, excluded from parse output entirely. MUST stay
# in sync with flaky_tests in the question-generation repos/astro/config.yaml.
# gen_report runs run/test/fix consistency checks over ALL parsed tests (not
# just F2P/P2P); a flaky flip in the fix phase would invalidate the report
# and zero a correct model patch (observed on nuxt pr-34320).
# Currently empty — populate together with config.yaml if stability runs
# surface flaky families.
_FLAKY_GLOBS: tuple = ()


def _is_flaky(name: str) -> bool:
    return any(fnmatch.fnmatchcase(name, g) for g in _FLAKY_GLOBS)


def _clean(name: str) -> str:
    name = _RE_RETRY_TIMING.sub("", name)
    name = _RE_RETRY_ONLY.sub("", name)
    name = _RE_JEST_TIMING.sub("", name)
    name = _RE_TIMING.sub("", name)
    return _RE_WS.sub(" ", name).strip()


@Instance.register("withastro", "astro")
class Astro(Instance):
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
        results: dict[str, str] = {}
        nt_stack: list[tuple[int, str]] = []  # open suites (indent, name)

        def nt_record(indent: int, name: str, status: str) -> None:
            while nt_stack and nt_stack[-1][0] >= indent:
                nt_stack.pop()
            full = " > ".join([s[1] for s in nt_stack] + [name])
            prev = results.get(full)
            if prev is None or _NT_SEVERITY[status] > _NT_SEVERITY[prev]:
                results[full] = status

        for raw_line in log.splitlines():
            line = _RE_ANSI.sub("", raw_line)
            if _RE_NT_SUMMARY.match(line):
                break
            m = _RE_NT_SUITE.match(line)
            if m:
                indent = len(m.group(1))
                while nt_stack and nt_stack[-1][0] >= indent:
                    nt_stack.pop()
                nt_stack.append((indent, _clean(m.group(2))))
                continue
            for regex, status in ((_RE_NT_PASS, "PASSED"),
                                  (_RE_NT_FAIL, "FAILED"),
                                  (_RE_NT_SKIP, "SKIPPED")):
                m = regex.match(line)
                if m:
                    indent, name = len(m.group(1)), _clean(m.group(2))
                    if nt_stack and nt_stack[-1] == (indent, name):
                        nt_stack.pop()       # suite-close line, not a test
                    else:
                        nt_record(indent, name, status)
                    break
        for name in [n for n in results if _is_flaky(n)]:
            del results[name]
        return mapping_to_testresult(results)
