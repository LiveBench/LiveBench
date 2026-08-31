"""Per-question timing weights for longest-first scheduling and time limits.

Agentic weights come from agentic_code_runner/data/question_weights.json
(median/p90 total_time_s per question across all models that record timing);
regenerate with scripts/build_question_weights.py. Regular categories have no
per-question timing, so ordering there uses a coarse hand-maintained per-task
table (evidence: the --regular-report output of the same script).

Longest-processing-time-first matters: the top-20 slowest of the 72 agentic
questions hold ~53% of total question-time, and simulated makespan shows
longest-first saves 19% of wall clock at parallelism 15 and 26% at 20.
"""

import functools
import json
from pathlib import Path

_WEIGHTS_PATH = Path(__file__).parent / "agentic_code_runner/data/question_weights.json"

# Fallback medians (seconds) for questions absent from the JSON, by language.
_LANG_DEFAULT_MEDIAN_S = {"typescript": 930, "javascript": 350, "python": 360}
_LANG_BY_PREFIX = {"tsab": "typescript", "jsab": "javascript", "pyab": "python"}

# Per-language agent time_limit (seconds). All languages keep the historical
# 5400 ceiling: the tighter js/py 4200 (chosen from healthy-serving p90s of
# ~2800/2450s) clipped a throttled-provider run into 31 empty-patch DNFs on
# 2026-08-31 — wall-clock caps are inference-compute budgets, and headroom
# computed on healthy serving does not survive rate-limit weather. Per-question
# overrides can still be added to the JSON as "time_limit_s".
_LANG_TIME_LIMIT_S = {"typescript": 5400, "javascript": 5400, "python": 5400}
_DEFAULT_TIME_LIMIT_S = 5400

# Regular categories: relative per-task weights (roughly median wall-clock
# minutes of a task's answer window across models). Only the ordering matters;
# unlisted tasks default to 5.
REGULAR_TASK_WEIGHTS = {
    "LCB_generation": 144,
    "zebra_puzzle_3": 37,
    "integrals_with_game": 34,
    "coding_completion": 32,
    "consecutive_events": 28,
    "math_comp": 17,
    "zebra_puzzle_2": 15,
    "olympiad": 15,
    "zebra_puzzle": 15,
    "math_comp_4": 15,
    "paraphrase_3": 14,
    "plot_unscrambling": 14,
    "olympiad_2": 14,
    "math_comp_2": 13,
    "web_of_lies_v2": 13,
    "sudoku": 13,
    "tablereformat": 12,
    "connections_3": 12,
    "simplify_3": 11,
    "story_generation_3": 11,
    "tablejoin_2": 10,
    "summarize_3": 10,
}
_DEFAULT_REGULAR_WEIGHT = 5


@functools.lru_cache(maxsize=1)
def load_agentic_weights() -> dict[str, dict]:
    try:
        with open(_WEIGHTS_PATH) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _language(question: dict) -> str | None:
    task = question.get("task")
    if task in _LANG_DEFAULT_MEDIAN_S:
        return task
    qid = str(question.get("question_id", ""))
    return _LANG_BY_PREFIX.get(qid.split("-", 1)[0])


def agentic_sort_key(question: dict) -> float:
    """Sort key for longest-first: negate to sort ascending, or use with reverse=True."""
    entry = load_agentic_weights().get(str(question.get("question_id")))
    if entry and entry.get("median_s"):
        return float(entry["median_s"])
    lang = _language(question)
    return float(_LANG_DEFAULT_MEDIAN_S.get(lang, max(_LANG_DEFAULT_MEDIAN_S.values())))


def agentic_time_limit(question: dict) -> int:
    """Per-question agent time_limit in seconds (JSON override, else per-language)."""
    entry = load_agentic_weights().get(str(question.get("question_id")))
    if entry and entry.get("time_limit_s"):
        return int(entry["time_limit_s"])
    lang = _language(question)
    return _LANG_TIME_LIMIT_S.get(lang, _DEFAULT_TIME_LIMIT_S)


def regular_task_weight(question: dict) -> float:
    task = question.get("task", "")
    # question tasks sometimes carry suffixes/prefixes; match the known key exactly
    # first, then by containment (e.g. "amps_hard_..." style variants).
    if task in REGULAR_TASK_WEIGHTS:
        return REGULAR_TASK_WEIGHTS[task]
    for key, w in REGULAR_TASK_WEIGHTS.items():
        if key in task:
            return w
    return _DEFAULT_REGULAR_WEIGHT
