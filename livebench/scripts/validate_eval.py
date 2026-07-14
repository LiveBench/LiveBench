#!/usr/bin/env python3
"""
Validate LiveBench evaluation results for a model.

Reports answer-generation errors, eval errors, and total error rates
per task, category, and overall. Returns exit code 1 if any category
exceeds the error threshold.

Only errors a retry can plausibly fix count toward the threshold:
transient API failures (rate limits, timeouts, dead sockets) and eval
infrastructure failures (docker errors, eval timeouts, instances that
never got evaluated). Terminal model failures — token exhaustion
(finish_reason=length), context-length overflows, content-policy
refusals — are genuine 0-scores: they are reported in their own column
but do not trigger retries.

Usage:
    python scripts/validate_eval.py --model minimax-m3
    python scripts/validate_eval.py --model minimax-m3 --threshold 5
    python scripts/validate_eval.py --model minimax-m3 --category agentic_coding_v2
"""

import argparse
import json
import glob
import os
import sys

CATEGORIES = [
    "coding", "data_analysis", "instruction_following",
    "language", "math", "reasoning", "agentic_coding", "agentic_coding_v2",
]

ANSWER_ERROR = "$ERROR$"

# Infrastructure failures only — things that prevent reliable scoring and
# that a retry can fix. Statuses match what the code actually emits:
#   eval_error         docker/harness infrastructure error (question skipped)
#   eval_timeout       eval container killed at the timeout
#   eval_flaky         legacy: ground-truth fix breaks other tests
#   not_submitted      instance missing from the eval report (scored 0!)
#   run_no_trajectory  inference produced no trajectory
#   run_error          inference harness error
# Excluded on purpose:
#   api_error          judgment echo of $ERROR$ in the answer file (double-count)
#   patch_error        model produced a malformed patch — wrong answer, not infra
#   empty_patch        model submitted no usable patch — wrong answer, not infra
#   token_exhaustion   model ran out of tokens — genuine failure, retry won't help
EVAL_FAILURE_STATUSES = {
    "eval_error", "eval_timeout", "eval_flaky",
    "not_submitted", "run_no_trajectory", "run_error",
}

# $ERROR$ answers whose error text matches any of these are terminal model
# failures: retrying reproduces them, so they must not gate retry rounds.
TERMINAL_ERROR_PATTERNS = [
    "finish_reason=length",
    "token_exhaustion",
    "context length",
    "context_length",
    "context window",
    "contextwindowexceeded",
    "maximum context",
    "prompt is too long",
    "exceeds the maximum",
    "content policy",
    "content_policy",
    "content_filter",
    "content management policy",
    "invalid prompt",
    "invalidprompt",
]

DETAIL_KEY_MAX = 60


def is_terminal_error(record):
    """True if this $ERROR$ answer is a genuine model failure, not infra."""
    text = f"{record.get('error', '')} {record.get('error_msg', '')}".lower()
    return any(p in text for p in TERMINAL_ERROR_PATTERNS)


def load_active_questions(task_dir):
    """Question ids that are currently live for a task, or None if unknown.

    Answer/judgment files keep records for questions that have since been
    removed from the benchmark; those are never re-run, so their errors must
    not count toward retry decisions (or the denominator).
    """
    qfile = os.path.join(task_dir, "question.jsonl")
    if not os.path.exists(qfile):
        return None
    active = set()
    for line in open(qfile):
        q = json.loads(line)
        if not q.get("livebench_removal_date"):
            active.add(q.get("question_id"))
    return active


def collect_answer_errors(data_root, model, categories):
    """Count $ERROR$ answers per task, split transient vs terminal.

    Answer files can accumulate records across reruns; only the latest
    record (by tstamp) per question is counted, so totals reflect the
    number of questions answered, not the number of attempts.
    """
    results = {}
    for cat in categories:
        pattern = os.path.join(data_root, cat, "*", "model_answer", f"{model}.jsonl")
        for f in sorted(glob.glob(pattern)):
            task = f.split(os.sep)[-3]
            key = f"{cat}/{task}"
            active = load_active_questions(os.path.dirname(os.path.dirname(f)))
            latest = {}
            for line in open(f):
                d = json.loads(line)
                qid = d.get("question_id")
                if active is not None and qid not in active:
                    continue
                prev = latest.get(qid)
                if prev is None or d.get("tstamp", 0) >= prev.get("tstamp", 0):
                    latest[qid] = d
            total = len(latest)
            errors = 0
            terminal = 0
            answer_details = {}
            for d in latest.values():
                if d["choices"][0]["turns"][0] == ANSWER_ERROR:
                    err_msg = str(d.get("error_msg") or d.get("error") or "unknown")[:DETAIL_KEY_MAX]
                    if is_terminal_error(d):
                        terminal += 1
                        err_msg = f"terminal: {err_msg}"
                    else:
                        errors += 1
                    answer_details[err_msg] = answer_details.get(err_msg, 0) + 1
            results[key] = {
                "total": total,
                "answer_errors": errors,
                "terminal_errors": terminal,
                "answer_details": answer_details,
            }
    return results


def collect_eval_errors(data_root, model, categories):
    """Count eval failures from judgment files.

    Judgment files accumulate records across reruns, so a question can have
    several entries for the same model. Only the latest record (by tstamp)
    per question reflects reality — a stale eval_timeout that was later
    re-judged cleanly must not count as an error.
    """
    results = {}
    for cat in categories:
        pattern = os.path.join(data_root, cat, "*", "model_judgment", "ground_truth_judgment.jsonl")
        for f in sorted(glob.glob(pattern)):
            task = f.split(os.sep)[-3]
            key = f"{cat}/{task}"
            active = load_active_questions(os.path.dirname(os.path.dirname(f)))
            latest = {}
            for line in open(f):
                d = json.loads(line)
                if d.get("model") != model:
                    continue
                qid = d.get("question_id")
                if active is not None and qid not in active:
                    continue
                prev = latest.get(qid)
                if prev is None or d.get("tstamp", 0) >= prev.get("tstamp", 0):
                    latest[qid] = d
            eval_errors = 0
            eval_details = {}
            for d in latest.values():
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
        term_err = a.get("terminal_errors", 0)
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
            "terminal_errors": term_err,
            "eval_errors": eval_err,
            "all_errors": all_err,
            "rate": rate,
            "details": details,
        }
    return merged


def print_report(merged, threshold):
    """Print the validation report."""
    fields = ["total", "answer_errors", "terminal_errors", "eval_errors", "all_errors"]

    # Task-level
    print(f"{'Task':<50} {'Total':>6} {'Ans':>5} {'Term':>5} {'Eval':>5} {'Retry':>5} {'Rate':>7}  Status")
    print("-" * 100)

    cat_totals = {}
    overall = {f: 0 for f in fields}
    needs_retry = False

    for key in sorted(merged):
        r = merged[key]
        cat = key.split("/")[0]
        status = "RETRY" if r["rate"] > threshold else "OK"

        if cat not in cat_totals:
            cat_totals[cat] = {f: 0 for f in fields}
            cat_totals[cat]["details"] = {}
        for field in fields:
            cat_totals[cat][field] += r[field]
            overall[field] += r[field]
        for k, v in r["details"].items():
            cat_totals[cat]["details"][k] = cat_totals[cat]["details"].get(k, 0) + v

        detail = ""
        if r["details"]:
            detail = " (" + ", ".join(f"{v}x {k}" for k, v in sorted(r["details"].items())) + ")"

        if r["all_errors"] > 0 or r["terminal_errors"] > 0:
            print(f"  {key:<48} {r['total']:>6} {r['answer_errors']:>5} {r['terminal_errors']:>5} {r['eval_errors']:>5} {r['all_errors']:>5} {r['rate']:>6.1f}%  {status}{detail}")

    # Category-level
    print()
    print(f"{'Category':<50} {'Total':>6} {'Ans':>5} {'Term':>5} {'Eval':>5} {'Retry':>5} {'Rate':>7}  Status")
    print("-" * 100)
    for cat in sorted(cat_totals):
        c = cat_totals[cat]
        rate = (c["all_errors"] / c["total"] * 100) if c["total"] > 0 else 0
        status = "RETRY" if rate > threshold else "OK"
        if rate > threshold:
            needs_retry = True
        detail = ""
        if c["details"]:
            detail = " (" + ", ".join(f"{v}x {k}" for k, v in sorted(c["details"].items())) + ")"
        print(f"  {cat:<48} {c['total']:>6} {c['answer_errors']:>5} {c['terminal_errors']:>5} {c['eval_errors']:>5} {c['all_errors']:>5} {rate:>6.1f}%  {status}{detail}")

    # Overall
    print()
    overall_rate = (overall["all_errors"] / overall["total"] * 100) if overall["total"] > 0 else 0
    print(f"  {'OVERALL':<48} {overall['total']:>6} {overall['answer_errors']:>5} {overall['terminal_errors']:>5} {overall['eval_errors']:>5} {overall['all_errors']:>5} {overall_rate:>6.1f}%")
    if overall["terminal_errors"] > 0:
        print()
        print(f"  Note: {overall['terminal_errors']} terminal failures (token exhaustion / context length / content policy)")
        print("  are genuine 0-scores and do not count toward the retry threshold.")

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
        print("RESULT: RETRY NEEDED — one or more categories exceed %.1f%% retryable error rate" % args.threshold)
        sys.exit(1)
    else:
        print("RESULT: ALL OK — all categories below %.1f%% retryable error rate" % args.threshold)
        sys.exit(0)


if __name__ == "__main__":
    main()
