"""Grade-as-you-go for the regular (non-agentic) categories.

GradingPool grades each answer right after gen_api_answer produces it, instead
of leaving all grading to the gen_ground_truth_judgment pass at the end. Each
work item carries its own (question, answer) pair — no qid-keyed lookup — so
question_id collisions across tasks cannot mispair (see the _answer_key fix in
the batch path). The batch judgment pass stays the idempotent backstop: run it
with --resume and it grades only what this pool didn't (keyed on answer_id).

Two lanes:
- a CPU-capped thread pool for normal graders (sympy, table compares, puzzles);
  grading is CPU-bound, and more workers than cores causes spurious timeout
  zeros (measured: integrals_with_game 37 vs true 76 at parallel 100 on 32
  cores), so the cap is a correctness guard, not a tuning knob;
- a single-worker lane for LCB-style code-execution tasks (coding_completion /
  LCB_generation), which fork test-runner subprocesses with wall-clock timeouts
  and historically required serial execution — the dedicated lane keeps them
  serial with respect to each other without serializing everything else the way
  the batch pass does.

Failure policy: anything that cannot be graded inline is simply left for the
backstop pass — a row is only written by play_a_match_gt itself, so a crash
here never writes a wrong judgment that --resume would then skip.
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor

# Tasks whose grading forks test-runner subprocesses; kept serial (see module doc).
SERIAL_GRADING_TASKS = {"coding_completion", "LCB_generation"}

# Mirrors play_a_match_gt's coding_test_case_tasks: these carry test cases
# instead of a ground_truth field.
_CODING_TEST_CASE_TASKS = {
    "coding_completion", "LCB_generation", "code_generation", "code_completion",
    "agentic_coding",
}

# Old-format instruction_following (graded batch-only, per-task, in the backstop
# pass); the boundary date matches gen_ground_truth_judgment.
_OLD_IF_BOUNDARY = "2025-11-25"


def resolve_grading_workers(requested: int | None) -> int:
    """-1/None = auto (CPU-capped); 0 = disabled; N = min(N, cpu_count)."""
    cores = os.cpu_count() or 8
    if requested is None or requested < 0:
        return min(cores, 16)
    return min(requested, cores)


class GradingPool:
    def __init__(self, model_id: str, workers: int, debug: bool = False):
        self.model_id = model_id
        self.debug = debug
        self._normal = ThreadPoolExecutor(max_workers=max(workers, 1),
                                          thread_name_prefix="grade")
        self._serial = ThreadPoolExecutor(max_workers=1, thread_name_prefix="grade-lcb")
        self._lock = threading.Lock()
        self._futures = []
        self._judgment_files: set[str] = set()
        self.graded = 0
        self.errors = 0
        self.skipped = 0

    def _route(self, question: dict) -> str | None:
        """Return 'serial'/'normal', or None to leave the answer to the backstop pass."""
        from livebench.common import AGENTIC_CODING_CATEGORIES
        category = question.get("category")
        task = question.get("task")
        if category in AGENTIC_CODING_CATEGORIES:
            return None  # own incremental pipeline in run_inference
        if category == "instruction_following" and \
                question.get("livebench_release_date", "") < _OLD_IF_BOUNDARY:
            return None  # old-format IF is graded per-task in batch only
        if (task not in _CODING_TEST_CASE_TASKS and category != "instruction_following"
                and "ground_truth" not in question):
            return None  # play_a_match_gt would raise; leave to the backstop
        if not question.get("_judgment_file"):
            return None
        return "serial" if task in SERIAL_GRADING_TASKS else "normal"

    def _grade(self, question: dict, answer: dict) -> None:
        from livebench.common import MatchSingle
        from livebench.gen_ground_truth_judgment import play_a_match_gt
        try:
            play_a_match_gt(
                MatchSingle(dict(question), self.model_id, answer),
                output_file=question["_judgment_file"],
                debug=self.debug,
            )
            with self._lock:
                self.graded += 1
        except Exception as e:
            with self._lock:
                self.errors += 1
            print(f"incremental judge: grading failed for {question.get('question_id')} "
                  f"({type(e).__name__}: {e}); the judgment sweep will cover it")

    def submit(self, question: dict, answer: dict | None) -> bool:
        """Queue one freshly generated answer for grading; False = left to the backstop."""
        if answer is None:
            return False
        lane = self._route(question)
        if lane is None:
            with self._lock:
                self.skipped += 1
            return False
        self._judgment_files.add(question["_judgment_file"])
        executor = self._serial if lane == "serial" else self._normal
        self._futures.append(executor.submit(self._grade, question, answer))
        return True

    def drain_and_close(self) -> tuple[int, int, int]:
        """Wait for queued grading to finish, tidy the judgment files, report counts."""
        self._normal.shutdown(wait=True)
        self._serial.shutdown(wait=True)
        from livebench.gen_ground_truth_judgment import reorg_output_file
        for f in self._judgment_files:
            try:
                reorg_output_file(f)
            except OSError as e:
                print(f"incremental judge: could not reorg {f}: {e}")
        print(f"incremental judge: done ({self.graded} graded inline, {self.errors} "
              f"failed, {self.skipped} left to the judgment sweep)")
        return self.graded, self.errors, self.skipped
