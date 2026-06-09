#!/usr/bin/env python3
"""
Validate LiveBench evaluation results for a model.

Reports answer-generation errors, eval errors, and total error rates
per task, category, and overall. Returns exit code 1 if any category
exceeds the error threshold.

Usage:
    python scripts/validate_eval.py --model minimax-m3
    python scripts/validate_eval.py --model minimax-m3 --threshold 5
    python scripts/validate_eval.py --model minimax-m3 --category agentic_coding
"""

import argparse
import json
import glob
import os
import sys

CATEGORIES = [
    "coding", "data_analysis", "instruction_following",
    "language", "math", "reasoning", "agentic_coding",
]

ANSWER_ERROR = "$ERROR$"

# Infrastructure failures only — things that prevent reliable scoring.
# api_error excluded: it's the judgment echo of $ERROR$ in the answer file.
# patch_error excluded: model produced a malformed patch — that's a wrong answer, not infra.
EVAL_FAILURE_STATUSES = {
    "eval_error", "eval_timeout",
    "eval_flaky", "eval_no_fix", "eval_missing_tests",
}


def collect_answer_errors(data_root, model, categories):
    """Count $ERROR$ answers per task."""
    results = {}
    for cat in categories:
        pattern = os.path.join(data_root, cat, "*", "model_answer", f"{model}.jsonl")
        for f in sorted(glob.glob(pattern)):
            task = f.split(os.sep)[-3]
            key = f"{cat}/{task}"
            total = 0
            errors = 0
            answer_details = {}
            for line in open(f):
                d = json.loads(line)
                total += 1
                if d["choices"][0]["turns"][0] == ANSWER_ERROR:
                    errors += 1
                    err_msg = d.get("error_msg", d.get("error", "unknown"))
                    answer_details[err_msg] = answer_details.get(err_msg, 0) + 1
            results[key] = {"total": total, "answer_errors": errors, "answer_details": answer_details}
    return results


def collect_eval_errors(data_root, model, categories):
    """Count eval failures from judgment files."""
    results = {}
    for cat in categories:
        pattern = os.path.join(data_root, cat, "*", "model_judgment", "ground_truth_judgment.jsonl")
        for f in sorted(glob.glob(pattern)):
            task = f.split(os.sep)[-3]
            key = f"{cat}/{task}"
            eval_errors = 0
            eval_details = {}
            for line in open(f):
                d = json.loads(line)
                if d.get("model") != model:
                    continue
                status = d.get("eval_status", "")
                if status in EVAL_FAILURE_STATUSES:
                    eval_errors += 1
                    eval_details[status] = eval_details.get(status, 0) + 1
            if key not in results:
                results[key] = {}
            results[key]["eval_errors"] = eval_errors
            results[key]["eval_details"] = eval_details
    return results


def merge_results(answer_results, eval_results):
    """Merge answer and eval error counts."""
    all_keys = sorted(set(list(answer_results.keys()) + list(eval_results.keys())))
    merged = {}
    for key in all_keys:
        a = answer_results.get(key, {})
        e = eval_results.get(key, {})
        total = a.get("total", 0)
        ans_err = a.get("answer_errors", 0)
        eval_err = e.get("eval_errors", 0)
        all_err = ans_err + eval_err
        rate = (all_err / total * 100) if total > 0 else 0
        details = {}
        for k, v in a.get("answer_details", {}).items():
            details[k] = details.get(k, 0) + v
        for k, v in e.get("eval_details", {}).items():
            details[k] = details.get(k, 0) + v
        merged[key] = {
            "total": total,
            "answer_errors": ans_err,
            "eval_errors": eval_err,
            "all_errors": all_err,
            "rate": rate,
            "details": details,
        }
    return merged


def print_report(merged, threshold):
    """Print the validation report."""
    # Task-level
    print(f"{'Task':<50} {'Total':>6} {'Ans':>5} {'Eval':>5} {'All':>5} {'Rate':>7}  Status")
    print("-" * 95)

    cat_totals = {}
    overall = {"total": 0, "answer_errors": 0, "eval_errors": 0, "all_errors": 0}
    needs_retry = False

    for key in sorted(merged):
        r = merged[key]
        cat = key.split("/")[0]
        status = "RETRY" if r["rate"] > threshold else "OK"

        if cat not in cat_totals:
            cat_totals[cat] = {"total": 0, "answer_errors": 0, "eval_errors": 0, "all_errors": 0, "details": {}}
        for field in ["total", "answer_errors", "eval_errors", "all_errors"]:
            cat_totals[cat][field] += r[field]
            overall[field] += r[field]
        for k, v in r["details"].items():
            cat_totals[cat]["details"][k] = cat_totals[cat]["details"].get(k, 0) + v

        detail = ""
        if r["details"]:
            detail = " (" + ", ".join(f"{v}x {k}" for k, v in sorted(r["details"].items())) + ")"

        if r["all_errors"] > 0:
            print(f"  {key:<48} {r['total']:>6} {r['answer_errors']:>5} {r['eval_errors']:>5} {r['all_errors']:>5} {r['rate']:>6.1f}%  {status}{detail}")

    # Category-level
    print()
    print(f"{'Category':<50} {'Total':>6} {'Ans':>5} {'Eval':>5} {'All':>5} {'Rate':>7}  Status")
    print("-" * 95)
    for cat in sorted(cat_totals):
        c = cat_totals[cat]
        rate = (c["all_errors"] / c["total"] * 100) if c["total"] > 0 else 0
        status = "RETRY" if rate > threshold else "OK"
        if rate > threshold:
            needs_retry = True
        detail = ""
        if c["details"]:
            detail = " (" + ", ".join(f"{v}x {k}" for k, v in sorted(c["details"].items())) + ")"
        print(f"  {cat:<48} {c['total']:>6} {c['answer_errors']:>5} {c['eval_errors']:>5} {c['all_errors']:>5} {rate:>6.1f}%  {status}{detail}")

    # Overall
    print()
    overall_rate = (overall["all_errors"] / overall["total"] * 100) if overall["total"] > 0 else 0
    print(f"  {'OVERALL':<48} {overall['total']:>6} {overall['answer_errors']:>5} {overall['eval_errors']:>5} {overall['all_errors']:>5} {overall_rate:>6.1f}%")

    return needs_retry


def main():
    parser = argparse.ArgumentParser(description="Validate LiveBench evaluation results")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--threshold", type=float, default=10.0, help="Error rate threshold (default: 10%%)")
    parser.add_argument("--category", nargs="+", default=None, help="Categories to check (default: all)")
    parser.add_argument("--data-root", default="data/live_bench", help="Path to data root")
    args = parser.parse_args()

    categories = args.category if args.category else CATEGORIES
    categories = [c for c in categories if os.path.isdir(os.path.join(args.data_root, c))]

    print(f"Validating {args.model} (threshold: {args.threshold}%)")
    print()

    answer_results = collect_answer_errors(args.data_root, args.model, categories)
    eval_results = collect_eval_errors(args.data_root, args.model, categories)
    merged = merge_results(answer_results, eval_results)

    needs_retry = print_report(merged, args.threshold)

    print()
    if needs_retry:
        print("RESULT: RETRY NEEDED — one or more categories exceed %.1f%% error rate" % args.threshold)
        sys.exit(1)
    else:
        print("RESULT: ALL OK — all categories below %.1f%% error rate" % args.threshold)
        sys.exit(0)


if __name__ == "__main__":
    main()
