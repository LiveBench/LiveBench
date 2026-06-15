"""Harness handler for `vuejs/core`.

This repo lives in TWO benchmarks and this single registered handler serves
both, dispatching on the question's `base.ref`:

  * LEGACY `agentic_coding` (typescript) questions — `base.ref` is a git ref
    (e.g. "main"). Built by CoreImageDefault (clone node:20 + run.sh/
    test-run.sh/fix-run.sh), parsed by the legacy vitest parser. UNCHANGED.

  * `agentic_coding_v2` (typescript_abacus) questions — `base.ref` is the full
    test command (contains whitespace, e.g. "./node_modules/.bin/vitest run
    --project unit --reporter=verbose"). Built by TypeScriptCustomBuildImage
    from prebuilt typescript_abacus/vuejs_m_core:pr-N images, parsed by the
    standard vitest verbose parser (kept in sync with parse_vitest_jest() in
    question_generation/agentic_coding_v2/typescript/scripts/04_validate_prs.py).

A git ref never contains whitespace; the v2 test command always does — that is
the discriminator (`_is_v2`).
"""

import fnmatch
import re
from typing import Optional, Union

from livebench.agentic_code_runner.eval.harness.image import (
    Config,
    File,
    Image,
    TypeScriptCustomBuildImage,
)
from livebench.agentic_code_runner.eval.harness.instance import Instance, TestResult
from livebench.agentic_code_runner.eval.harness.pull_request import PullRequest
from livebench.agentic_code_runner.eval.harness.test_result import mapping_to_testresult


# ── v2 vitest verbose parser (in sync with 04_validate_prs.parse_vitest_jest) ──
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

# Known-flaky families, excluded from v2 parse output entirely. MUST stay in
# sync with flaky_tests in question_generation .../repos/core/config.yaml.
# Both are CPU-contention-sensitive (a perf assertion + a heavily-async
# Suspense test) — they flip under load and would trip gen_report's
# run/test/fix consistency check, zeroing a correct patch.
_FLAKY_GLOBS: tuple = (
    "*should not lead to exponential perf cost*",
    "*reactivity/computed > performance when removing dependencies from deeply nested computeds",
    "*Suspense > combined usage (nested async + nested suspense + multiple deps)",
    "*Suspense > nested suspense (child resolves first)",
    "*Suspense > branch switch to 3rd branch before resolve",
    "*Suspense > nested suspense (w/ suspensible) switch several times before parent suspense resolve",
    "*SFC style preprocessors > scss @import",
)


def _is_flaky(name: str) -> bool:
    return any(fnmatch.fnmatchcase(name, g) for g in _FLAKY_GLOBS)


def _clean_v2(name: str) -> str:
    name = _RE_RETRY_TIMING.sub("", name)
    name = _RE_RETRY_ONLY.sub("", name)
    name = _RE_JEST_TIMING.sub("", name)
    name = _RE_TIMING.sub("", name)
    name = _RE_WS.sub(" ", name).strip()
    return _RE_PROJ_PIPE.sub(r"\1 ", name)


# ── Legacy agentic_coding image classes (UNCHANGED) ───────────────────────────
class CoreImageBase(Image):
    def __init__(self, pr: PullRequest, config: Config):
        self._pr = pr
        self._config = config

    @property
    def pr(self) -> PullRequest:
        return self._pr

    @property
    def config(self) -> Config:
        return self._config

    def dependency(self) -> Union[str, "Image"]:
        return "node:20"

    def image_tag(self) -> str:
        return "base"

    def workdir(self) -> str:
        return "base"

    def files(self) -> list[File]:
        return []

    def dockerfile(self) -> str:
        image_name = self.dependency()
        if isinstance(image_name, Image):
            image_name = image_name.image_full_name()

        if self.config.need_clone:
            code = f"RUN git clone https://github.com/{self.pr.org}/{self.pr.repo}.git /home/{self.pr.repo}"
        else:
            code = f"COPY {self.pr.repo} /home/{self.pr.repo}"

        return f"""FROM {image_name}

{self.global_env}

WORKDIR /home/

RUN apt update && apt install -y git
RUN npm install -g pnpm

{code}

{self.clear_env}

"""


class CoreImageDefault(Image):
    def __init__(self, pr: PullRequest, config: Config):
        self._pr = pr
        self._config = config

    @property
    def pr(self) -> PullRequest:
        return self._pr

    @property
    def config(self) -> Config:
        return self._config

    def dependency(self) -> Image | None:
        return CoreImageBase(self.pr, self.config)

    def image_tag(self) -> str:
        return f"pr-{self.pr.number}"

    def workdir(self) -> str:
        return f"pr-{self.pr.number}"

    def files(self) -> list[File]:
        return [
            File(
                ".",
                "fix.patch",
                f"{self.pr.fix_patch}",
            ),
            File(
                ".",
                "test.patch",
                f"{self.pr.test_patch}",
            ),
            File(
                ".",
                "check_git_changes.sh",
                """#!/bin/bash
set -e

if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
  echo "check_git_changes: Not inside a git repository"
  exit 1
fi

if [[ -n $(git status --porcelain) ]]; then
  echo "check_git_changes: Uncommitted changes"
  exit 1
fi

echo "check_git_changes: No uncommitted changes"
exit 0

""".format(
                    pr=self.pr
                ),
            ),
            File(
                ".",
                "prepare.sh",
                """#!/bin/bash
set -e

cd /home/{pr.repo}
git reset --hard
bash /home/check_git_changes.sh
git checkout {pr.base.sha}
bash /home/check_git_changes.sh

pnpm install || true

""".format(
                    pr=self.pr
                ),
            ),
            File(
                ".",
                "run.sh",
                """#!/bin/bash
set -e

cd /home/{pr.repo}
pnpm run test-unit --no-watch --reporter=verbose

""".format(
                    pr=self.pr
                ),
            ),
            File(
                ".",
                "test-run.sh",
                """#!/bin/bash
set -e

cd /home/{pr.repo}
git apply /home/test.patch
pnpm run test-unit --no-watch --reporter=verbose

""".format(
                    pr=self.pr
                ),
            ),
            File(
                ".",
                "fix-run.sh",
                """#!/bin/bash
set -e

cd /home/{pr.repo}
git apply /home/test.patch /home/fix.patch
pnpm run test-unit --no-watch --reporter=verbose

""".format(
                    pr=self.pr
                ),
            ),
        ]

    def dockerfile(self) -> str:
        image = self.dependency()
        name = image.image_name()
        tag = image.image_tag()

        copy_commands = ""
        for file in self.files():
            copy_commands += f"COPY {file.name} /home/\n"

        prepare_commands = "RUN bash /home/prepare.sh"

        return f"""FROM {name}:{tag}

{self.global_env}

{copy_commands}

{prepare_commands}

{self.clear_env}

"""


def _parse_log_legacy(test_log: str) -> TestResult:
    passed_tests = set()
    failed_tests = set()
    skipped_tests = set()

    re_pass = re.compile(r"^ ?[✓√] (.+)$")
    re_fail = re.compile(r"^ ?× (.+)$")
    re_timing = re.compile(r" \d+(?:\.\d+)?(?:ms|s)$")

    for line in test_log.splitlines():
        pass_match = re_pass.match(line)
        if pass_match:
            test = re_timing.sub("", pass_match.group(1)).rstrip()
            passed_tests.add(test)

        fail_match = re_fail.match(line)
        if fail_match:
            test = re_timing.sub("", fail_match.group(1)).rstrip()
            failed_tests.add(test)

    return TestResult(
        passed_count=len(passed_tests),
        failed_count=len(failed_tests),
        skipped_count=len(skipped_tests),
        passed_tests=passed_tests,
        failed_tests=failed_tests,
        skipped_tests=skipped_tests,
    )


@Instance.register("vuejs", "core")
class Core(Instance):
    def __init__(self, pr: PullRequest, config: Config, *args, **kwargs):
        super().__init__()
        self._pr = pr
        self._config = config

    @property
    def pr(self) -> PullRequest:
        return self._pr

    def _is_v2(self) -> bool:
        """Discriminate agentic_coding_v2 from the legacy agentic_coding
        benchmark by PR number. v2 questions were collected from the 2026
        window (vue PRs >= 14475); every legacy benchmark question is from the
        old set (vue PRs <= 11517). PR number is the one identifier reliably
        populated in BOTH the image-build and gen_report parse code paths —
        base.ref is NOT (it carries the test command when fix-run.sh is
        generated but a bare git ref when gen_report re-parses, which silently
        routed v2 logs to the legacy parser and scored every question 0)."""
        try:
            return int(self.pr.number) >= 12000
        except (TypeError, ValueError, AttributeError):
            # Fallback: v2 carries the test command (whitespace) in base.ref.
            return " " in (self.pr.base.ref or "").strip()

    def dependency(self) -> Optional[Image]:
        if self._is_v2():
            return TypeScriptCustomBuildImage(
                self.pr, self._config, base_prefix="typescript_abacus")
        return CoreImageDefault(self.pr, self._config)

    def run(self, run_cmd: str = "") -> str:
        if run_cmd:
            return run_cmd
        return "bash /home/run.sh"

    def test_patch_run(self, test_patch_run_cmd: str = "") -> str:
        if test_patch_run_cmd:
            return test_patch_run_cmd
        return "bash /home/test-run.sh"

    def fix_patch_run(self, fix_patch_run_cmd: str = "") -> str:
        if fix_patch_run_cmd:
            return fix_patch_run_cmd
        return "bash /home/fix-run.sh"

    def parse_log(self, test_log: str) -> TestResult:
        if not self._is_v2():
            return _parse_log_legacy(test_log)
        # v2: standard vitest verbose parser
        test_status_map: dict = {}
        for raw_line in test_log.splitlines():
            line = _RE_ANSI.sub("", raw_line)
            m = _RE_PASS.match(line)
            if m:
                test_status_map[_clean_v2(m.group(1))] = "PASSED"
                continue
            m = _RE_FAIL.match(line)
            if m:
                test_status_map[_clean_v2(m.group(1))] = "FAILED"
                continue
            m = _RE_SKIP.match(line)
            if m:
                test_status_map[_clean_v2(m.group(1))] = "SKIPPED"
        for name in [n for n in test_status_map if _is_flaky(n)]:
            del test_status_map[name]
        return mapping_to_testresult(test_status_map)
