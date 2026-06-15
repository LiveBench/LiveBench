#!/usr/bin/env python3
"""Report LiveBench eval progress per category.

For a model, counts against the total questions in each category how many
answers were submitted, how many ran but produced nothing, and how many hit
an API error (broken down by type, e.g. RateLimitError), plus how many have
been judged. Prints it as a table, a single heartbeat line, or a markdown
summary with progress bars for the GitHub Actions UI.

The same shape covers agentic coding: a submission there is the patch the
agent produced, an empty answer is a run that finished without one.
"""

import argparse
import glob
import json
import os
from collections import Counter
from datetime import datetime

CATEGORIES = [
    "coding", "data_analysis", "instruction_following",
    "language", "math", "reasoning", "agentic_coding",
]

ANSWER_ERROR = "$ERROR$"


def count_lines(path):
    try:
        with open(path) as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def read_jsonl(path):
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    except OSError:
        return


def answer_turn(answer):
    turns = answer["choices"][0]["turns"]
    return turns[0] if isinstance(turns, list) else turns


def category_progress(data_root, model, category):
    p = {"total": 0, "answers": 0, "submitted": 0, "empty": 0,
         "errors": 0, "judged": 0, "error_types": Counter()}
    for task in sorted(glob.glob(os.path.join(data_root, category, "*"))):
        if not os.path.isdir(task):
            continue
        p["total"] += count_lines(os.path.join(task, "question.jsonl"))
        for answer in read_jsonl(os.path.join(task, "model_answer", f"{model}.jsonl")):
            p["answers"] += 1
            turn = answer_turn(answer)
            if turn == ANSWER_ERROR:
                p["errors"] += 1
                p["error_types"][answer.get("error") or "unknown"] += 1
            elif str(turn).strip():
                p["submitted"] += 1
            else:
                p["empty"] += 1
        for judgment in read_jsonl(os.path.join(task, "model_judgment", "ground_truth_judgment.jsonl")):
            if judgment.get("model") == model:
                p["judged"] += 1
    return p


def totals(progress):
    keys = ("total", "answers", "submitted", "empty", "errors", "judged")
    return {key: sum(p[key] for p in progress.values()) for key in keys}


def error_breakdown(progress):
    counts = Counter()
    for p in progress.values():
        counts.update(p["error_types"])
    return ", ".join(f"{kind} {n}" for kind, n in counts.most_common())


def percent(done, total):
    return round(100 * done / total) if total else 0


def status(p):
    if p["total"] == 0:
        return "-"
    if p["judged"] >= p["total"]:
        return "done"
    if p["answers"] >= p["total"]:
        return "grading"
    if p["answers"]:
        return "running"
    return "waiting"


def bar(done, total, width=10):
    filled = round(width * done / total) if total else 0
    return "█" * filled + "░" * (width - filled)


def render_line(progress):
    parts = []
    for category, p in progress.items():
        if not p["total"]:
            continue
        label = "done" if p["judged"] >= p["total"] else f"{p['submitted']}/{p['total']}"
        parts.append(f"{category} {label}")
    t = totals(progress)
    tail = f"{t['submitted']} submitted"
    if t["errors"]:
        tail += f", {t['errors']} errors"
    if t["empty"]:
        tail += f", {t['empty']} no submission"
    tail += f", {percent(t['judged'], t['total'])}% judged"
    return f"{datetime.now():%H:%M}  " + "   ".join(parts) + f"   ({tail})"


def render_markdown(progress, model, round_label):
    title = f"### {model} progress"
    if round_label:
        title += f" — round {round_label}"
    rows = [title, f"_updated {datetime.now():%H:%M:%S}_", "",
            "| category | submitted | no submission | errors | judged | status |",
            "| --- | --- | --- | --- | --- | --- |"]
    for category, p in progress.items():
        if not p["total"]:
            continue
        rows.append(f"| {category} "
                    f"| {bar(p['submitted'], p['total'])} {p['submitted']}/{p['total']} "
                    f"| {p['empty'] or ''} "
                    f"| {p['errors'] or ''} "
                    f"| {bar(p['judged'], p['total'])} {p['judged']}/{p['total']} "
                    f"| {status(p)} |")
    t = totals(progress)
    rows.append(f"| **total** "
                f"| **{t['submitted']}/{t['total']} ({percent(t['submitted'], t['total'])}%)** "
                f"| **{t['empty'] or ''}** | **{t['errors'] or ''}** "
                f"| **{t['judged']}/{t['total']} ({percent(t['judged'], t['total'])}%)** | |")
    errors = error_breakdown(progress)
    if errors:
        rows += ["", f"errors: {errors}"]
    return "\n".join(rows)


def render_table(progress):
    rows = [f"{'category':<24}{'submitted':>13}{'empty':>7}{'errors':>8}{'judged':>13}  status"]
    for category, p in progress.items():
        if not p["total"]:
            continue
        submitted = f"{p['submitted']}/{p['total']}"
        judged = f"{p['judged']}/{p['total']}"
        rows.append(f"{category:<24}{submitted:>13}{p['empty']:>7}{p['errors']:>8}{judged:>13}  {status(p)}")
    errors = error_breakdown(progress)
    if errors:
        rows += ["", f"errors: {errors}"]
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description="Report LiveBench eval progress per category")
    parser.add_argument("--model", required=True)
    parser.add_argument("--category", nargs="+", default=CATEGORIES)
    parser.add_argument("--data-root", default="data/live_bench")
    parser.add_argument("--round")
    parser.add_argument("--oneline", action="store_true")
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    progress = {c: category_progress(args.data_root, args.model, c) for c in args.category}

    if args.oneline:
        print(render_line(progress))
    elif args.markdown:
        print(render_markdown(progress, args.model, args.round))
    else:
        print(render_table(progress))


if __name__ == "__main__":
    main()
