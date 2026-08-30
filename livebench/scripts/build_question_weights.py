"""Regenerate agentic_code_runner/data/question_weights.json from the corpus.

Aggregates per-question median/p90 of `total_time_s` over every model answer file
that records timing (roughly a third of models do), for use by
livebench.question_weights (longest-first scheduling and per-language time
limits). Run from the livebench/ directory:

    python scripts/build_question_weights.py [--data-dir data/live_bench/agentic_coding_v2] [--min-obs 5]

Also prints a coarse per-task wall-clock table for the regular categories
(computed from answer-row tstamp spread per task, the only timing regular rows
carry) as a reference for question_weights.REGULAR_TASK_WEIGHTS — that table is
maintained by hand, this output is just evidence for updating it.
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

LANG_TASKS = ("python", "javascript", "typescript")


def collect_agentic(data_dir: Path, min_obs: int) -> dict:
    times: dict[str, list[float]] = defaultdict(list)
    lang: dict[str, str] = {}
    for task in LANG_TASKS:
        qfile = data_dir / task / "question.jsonl"
        if qfile.exists():
            for line in open(qfile):
                q = json.loads(line)
                lang[str(q["question_id"])] = task
        for ans_file in sorted((data_dir / task / "model_answer").glob("*.jsonl")):
            for line in open(ans_file):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = row.get("total_time_s")
                if not t or t <= 0:
                    continue
                qid = str(row["question_id"])
                lang.setdefault(qid, task)
                times[qid].append(float(t))

    weights = {}
    skipped = []
    for qid, ts in sorted(times.items()):
        if len(ts) < min_obs:
            skipped.append(qid)
            continue
        ts.sort()
        weights[qid] = {
            "median_s": round(statistics.median(ts)),
            "p90_s": round(ts[min(len(ts) - 1, int(0.9 * len(ts)))]),
            "n_obs": len(ts),
            "lang": lang.get(qid, "unknown"),
        }
    if skipped:
        print(f"skipped {len(skipped)} questions with < {min_obs} observations: {skipped}")
    return weights


def print_regular_task_windows(live_bench_dir: Path) -> None:
    """Per-task tstamp spread (max-min over a model's rows), medianed across models.

    Regular answer rows carry no per-question duration, so the per-model wall-clock
    window of each task is the best available signal for which tasks are long.
    """
    windows: dict[str, list[float]] = defaultdict(list)
    for cat_dir in sorted(live_bench_dir.iterdir()):
        if not cat_dir.is_dir() or cat_dir.name.startswith("agentic_coding"):
            continue
        for task_dir in sorted(cat_dir.iterdir()):
            ans_dir = task_dir / "model_answer"
            if not ans_dir.is_dir():
                continue
            task_key = f"{cat_dir.name}/{task_dir.name}"
            for ans_file in ans_dir.glob("*.jsonl"):
                stamps = []
                for line in open(ans_file):
                    try:
                        stamps.append(float(json.loads(line).get("tstamp") or 0))
                    except (json.JSONDecodeError, TypeError):
                        continue
                stamps = [s for s in stamps if s > 0]
                if len(stamps) >= 5:
                    windows[task_key].append(max(stamps) - min(stamps))
    print("\n# Regular per-task wall-clock windows (median minutes across models);")
    print("# evidence for question_weights.REGULAR_TASK_WEIGHTS, top 20:")
    ranked = sorted(windows.items(), key=lambda kv: -statistics.median(kv[1]))
    for task, ws in ranked[:20]:
        print(f"{statistics.median(ws) / 60:8.1f}  {task}  (n={len(ws)})")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data/live_bench/agentic_coding_v2"))
    parser.add_argument("--min-obs", type=int, default=5)
    parser.add_argument(
        "--output", type=Path,
        default=Path("agentic_code_runner/data/question_weights.json"))
    parser.add_argument("--regular-report", action="store_true",
                        help="Also print the regular-category per-task window table")
    args = parser.parse_args()

    weights = collect_agentic(args.data_dir, args.min_obs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(weights, f, indent=2, sort_keys=True)
        f.write("\n")
    by_lang: dict[str, list[int]] = defaultdict(list)
    for w in weights.values():
        by_lang[w["lang"]].append(w["median_s"])
    print(f"wrote {len(weights)} question weights to {args.output}")
    for lg, meds in sorted(by_lang.items()):
        print(f"  {lg}: {len(meds)} questions, median-of-medians {statistics.median(meds):.0f}s")

    if args.regular_report:
        print_regular_task_windows(args.data_dir.parent)


if __name__ == "__main__":
    main()
