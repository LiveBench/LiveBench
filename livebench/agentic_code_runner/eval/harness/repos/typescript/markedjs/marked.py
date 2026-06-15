"""Harness handler for `markedjs/marked` (typescript_abacus).

Test runner: Node's built-in node:test with the spec reporter
(`node --test --test-concurrency=1 --test-reporter=spec`). Output:
  "▶ Marked"                              suite open
  "  ✔ should work with use (0.65ms)"     test pass
  "  ✖ should fail (1.2ms)"               test fail
  "  ﹣ skipped one (0.1ms) # SKIP"        test skip
  "✔ Marked (18.86ms)"                    suite close — same glyph as a
                                          passing test; disambiguated by
                                          matching the open-suite stack.
Test names are the full suite path joined with " > ". marked's CommonMark
and GFM spec suites register identically-named tests whose completion order
is non-deterministic, so duplicates merge worst-status-wins (order-
independent). Parsing stops at the "ℹ tests" summary because node re-prints
failing leaf lines below it.

The regexes and the parsing logic MUST stay in sync with
parse_vitest_jest() in question_generation/agentic_coding_v2/typescript/
scripts/04_validate_prs.py.
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


def _clean(name: str) -> str:
    name = _RE_RETRY_TIMING.sub("", name)
    name = _RE_RETRY_ONLY.sub("", name)
    name = _RE_JEST_TIMING.sub("", name)
    name = _RE_TIMING.sub("", name)
    return _RE_WS.sub(" ", name).strip()


@Instance.register("markedjs", "marked")
class Marked(Instance):
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
        return mapping_to_testresult(results)
