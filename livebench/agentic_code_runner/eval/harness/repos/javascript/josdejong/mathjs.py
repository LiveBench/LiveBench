"""Harness handler for `josdejong/mathjs` (agentic_coding_v2 / javascript).

Test runner: Mocha 11 (spec reporter), `node ./node_modules/.bin/mocha
test/unit-tests --reporter spec`. mathjs is `type: module`; the unit tests
`import` straight from `src/*.js`, so node runs the ESM source natively — NO
build step (the repo's `.babelrc` is dist-only). Fresh repo (no legacy handler)
→ always builds from the pre-built per-PR image via TypeScriptCustomBuildImage.

Mocha 11 spec output (non-TTY → ANSI stripped here):
    add                              describe header (plain indented text)
      ✔ should add two numbers       pass  (mocha 10/11 glyph is ✔ U+2714)
      2) should reject bad input      fail  (cumulative N) prefix, in-tree)
      - skipped one                   pending/skip
    6652 passing (10s)               summary — parsing STOPS here

QUALIFIED NAMES: unlike validator.js / ajv (bare leaf titles), mathjs keys each
test by its full describe chain joined with ' > ' (e.g. "add > should throw …").
The suite has 451 leaf titles that repeat across files — one ("should throw an
error in case of invalid number of arguments") appears 79× — which bare-title
keying would collapse under worst-status-wins and confound FAIL_TO_PASS.
Qualifying cuts collisions to 38 (genuine same-describe dupes, handled by
worst-status-wins). This MUST stay byte-identical to parse_mocha() with
`mocha_qualified_names: true` in
question_generation/agentic_coding_v2/javascript/scripts/04_validate_prs.py.
The describe chain is reconstructed from leading-whitespace indentation (the
spec reporter indents 2 spaces per nesting level); test lines pop the stack to
their ancestors before joining.
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


@Instance.register("josdejong", "mathjs")
class Mathjs(Instance):
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

        # describe-chain stack: (indent, title). Qualifies each leaf title.
        stack: list[tuple[int, str]] = []

        def qualify(line: str, leaf: str) -> str:
            indent = len(line) - len(line.lstrip())
            while stack and stack[-1][0] >= indent:
                stack.pop()
            prefix = " > ".join(t for _, t in stack)
            return f"{prefix} > {leaf}" if prefix else leaf

        for raw_line in log.splitlines():
            line = _RE_M_ANSI.sub("", raw_line)
            if _RE_M_SUMMARY.match(line):
                break
            m = _RE_M_PASS.match(line)
            if m:
                record(qualify(line, m.group(1)), "PASSED")
                continue
            m = _RE_M_FAIL.match(line)
            if m:
                record(qualify(line, m.group(1)), "FAILED")
                continue
            m = _RE_M_PEND.match(line)
            if m:
                record(qualify(line, m.group(1)), "SKIPPED")
                continue
            # A non-empty line that is none of the above is a describe header.
            # Real mocha spec describe headers are indented >= 2 spaces; ignore
            # column-0 noise. The grade-side fix-run.sh runs under `set -x`,
            # which echoes "+ node …/mocha …" at indent 0; npm warnings are also
            # at column 0. Left unfiltered such a line becomes a phantom root
            # describe and prefixes every test name (→ all P2P "missing" at
            # grade time). MUST match the same guard in 04_validate_prs.py.
            if line.strip():
                indent = len(line) - len(line.lstrip())
                if indent >= 2:
                    while stack and stack[-1][0] >= indent:
                        stack.pop()
                    stack.append((indent, line.strip()))
        return mapping_to_testresult(results)
