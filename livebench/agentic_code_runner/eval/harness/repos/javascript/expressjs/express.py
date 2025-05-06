import re
from dataclasses import asdict, dataclass
from json import JSONDecoder
from typing import Generator, Optional, Union

from dataclasses_json import dataclass_json

from multi_swe_bench.harness.image import Config, File, Image
from multi_swe_bench.harness.instance import Instance, TestResult
from multi_swe_bench.harness.pull_request import PullRequest


class ImageBase(Image):
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
        return "ubuntu:latest"

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
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
RUN apt update && apt install -y git nodejs npm

{code}

{self.clear_env}

"""


class ImageDefault(Image):
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
        return ImageBase(self.pr, self._config)

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

npm install || true

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
npm run test-ci -- --reporter json
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
git apply --whitespace=nowarn /home/test.patch
npm run test-ci -- --reporter json 

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
git apply --whitespace=nowarn /home/test.patch /home/fix.patch
npm run test-ci -- --reporter json

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


@Instance.register("expressjs", "express")
class Express(Instance):
    def __init__(self, pr: PullRequest, config: Config, *args, **kwargs):
        super().__init__()
        self._pr = pr
        self._config = config

    @property
    def pr(self) -> PullRequest:
        return self._pr

    def dependency(self) -> Optional[Image]:
        return ImageDefault(self.pr, self._config)

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
        if self.pr.number not in [4212, 4852, 5555, 5841]:
            return self.parse_json_log(test_log)

        passed_tests = set()
        failed_tests = set()
        skipped_tests = set()

        re_pass_test = re.compile(r"^✔ (.+?)(?:\s\(\d*\.?\d*\s*\w+\))?$")
        re_fail_test = re.compile(r"^\d+\) (.+?)(?:\s\(\d*\.?\d*\s*\w+\))?$")

        for line in test_log.splitlines():
            line = line.strip()
            if not line:
                continue

            pass_test_match = re_pass_test.match(line)
            if pass_test_match:
                test = pass_test_match.group(1)
                passed_tests.add(test)

            fail_test_match = re_fail_test.match(line)
            if fail_test_match:
                test = fail_test_match.group(1)
                failed_tests.add(test)

        return TestResult(
            passed_count=len(passed_tests),
            failed_count=len(failed_tests),
            skipped_count=len(skipped_tests),
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
        )

    def parse_json_log(self, test_log: str) -> TestResult:
        passed_tests: set[str] = set()
        failed_tests: set[str] = set()
        skipped_tests: set[str] = set()

        @dataclass_json
        @dataclass
        class ExpressStats:
            suites: int
            tests: int
            passes: int
            pending: int
            failures: int
            start: str
            end: str
            duration: int

            @classmethod
            def from_dict(cls, d: dict) -> "TestResult":
                return cls(**d)

            @classmethod
            def from_json(cls, json_str: str) -> "TestResult":
                return cls.from_dict(cls.schema().loads(json_str))

            def dict(self) -> dict:
                return asdict(self)

            def json(self) -> str:
                return self.to_json(ensure_ascii=False)

        @dataclass_json
        @dataclass
        class ExpressTest:
            title: str
            fullTitle: str
            currentRetry: int
            err: dict
            duration: Optional[int] = None
            speed: Optional[str] = None

            @classmethod
            def from_dict(cls, d: dict) -> "TestResult":
                return cls(**d)

            @classmethod
            def from_json(cls, json_str: str) -> "TestResult":
                return cls.from_dict(cls.schema().loads(json_str))

            def dict(self) -> dict:
                return asdict(self)

            def json(self) -> str:
                return self.to_json(ensure_ascii=False)

        @dataclass_json
        @dataclass
        class ExpressInfo:
            stats: ExpressStats
            tests: list[ExpressTest]
            pending: list[ExpressTest]
            failures: list[ExpressTest]
            passes: list[ExpressTest]

            @classmethod
            def from_dict(cls, d: dict) -> "ExpressInfo":
                return cls(**d)

            @classmethod
            def from_json(cls, json_str: str) -> "ExpressInfo":
                return cls.from_dict(cls.schema().loads(json_str))

            def dict(self) -> dict:
                return asdict(self)

            def json(self) -> str:
                return self.to_json(ensure_ascii=False)

        def extract_json_objects(
            text: str, decoder=JSONDecoder()
        ) -> Generator[dict, None, None]:
            pos = 0
            while True:
                match = text.find("{", pos)
                if match == -1:
                    break
                try:
                    result, index = decoder.raw_decode(text[match:])
                    yield result
                    pos = match + index
                except ValueError:
                    pos = match + 1

        pattern = r"^(=+|Writing .*)$"
        test_log = re.sub(pattern, "", test_log, flags=re.MULTILINE)

        test_log = test_log.replace("\r\n", "")
        test_log = test_log.replace("\n", "")

        for obj in extract_json_objects(test_log):
            info = ExpressInfo.from_dict(obj)
            for test in info.passes:
                passed_tests.add(f"{test.fullTitle}")
            for test in info.failures:
                failed_tests.add(f"{test.fullTitle}")
            for test in info.pending:
                skipped_tests.add(f"{test.fullTitle}")

        for test in failed_tests:
            if test in passed_tests:
                passed_tests.remove(test)
            if test in skipped_tests:
                skipped_tests.remove(test)

        for test in skipped_tests:
            if test in passed_tests:
                passed_tests.remove(test)

        return TestResult(
            passed_count=len(passed_tests),
            failed_count=len(failed_tests),
            skipped_count=len(skipped_tests),
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
        )
