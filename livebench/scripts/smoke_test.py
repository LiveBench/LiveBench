#!/usr/bin/env python3
"""
Smoke test for LiveBench model evaluation.

Runs 1 question, verifies:
  1. Model config exists and loads
  2. API call succeeds (not $ERROR$)
  3. Token counts are populated
  4. Cost tracking works
  5. Grading produces a score

Usage:
    python scripts/smoke_test.py --model minimax-m3
"""

import argparse
import json
import os
import sys
import subprocess


def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAIL: command returned {result.returncode}")
        print(result.stderr[-500:] if result.stderr else "")
        return None
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description="Smoke test for LiveBench")
    parser.add_argument("--model", required=True)
    parser.add_argument("--bench", default="live_bench/math/AMPS_Hard")
    args = parser.parse_args()

    model = args.model
    bench = args.bench
    task = bench.split("/")[-1]
    answer_file = f"data/{bench}/model_answer/{model}.jsonl"
    errors = []

    print(f"Smoke testing {model}")
    print()

    # 1. Model config
    print("1. Model config...", end=" ")
    out = run_cmd(f'python -c "from livebench.model import get_model_config; c = get_model_config(\'{model}\'); print(c.display_name, list(c.api_name.keys())[0])"')
    if out is None:
        errors.append("Model config not found or failed to load")
        print("FAIL")
    else:
        print(f"OK ({out.strip()})")

    # 2. API call — delete existing answer first to force fresh call
    print("2. API call...", end=" ")
    # Remove existing answers for this question to force a fresh call
    if os.path.exists(answer_file):
        os.remove(answer_file)

    out = run_cmd(
        f"python gen_api_answer.py --model {model} --question-source jsonl "
        f"--bench-name {bench} --question-end 1 --use-litellm 2>&1"
    )
    if out is None or not os.path.exists(answer_file):
        errors.append("API call failed — no answer file produced")
        print("FAIL")
        print("\n".join(f"  {e}" for e in errors))
        sys.exit(1)

    answer = json.loads(open(answer_file).readline())
    content = answer["choices"][0]["turns"][0]

    if content == "$ERROR$":
        err_msg = answer.get("error_msg", answer.get("error", "unknown"))
        errors.append(f"API returned $ERROR$: {err_msg}")
        print(f"FAIL ({err_msg})")
    else:
        print(f"OK ({len(content)} chars)")

    # 3. Token counts
    print("3. Token tracking...", end=" ")
    out_tok = answer.get("total_output_tokens", 0) or 0
    in_tok = answer.get("total_input_tokens", 0) or 0
    if out_tok == 0 and content != "$ERROR$":
        errors.append("Output tokens = 0 for non-error answer")
        print(f"FAIL (out={out_tok}, in={in_tok})")
    else:
        print(f"OK (in={in_tok}, out={out_tok})")

    # 4. Cost tracking (warning only — inline configs may not have cost_per_million)
    print("4. Cost tracking...", end=" ")
    cost = answer.get("cost_usd")
    if cost is None or cost == 0:
        print(f"WARN (cost not tracked — add cost_per_million to config)")
    else:
        print(f"OK (${cost:.4f})")

    # 5. Grading
    print("5. Grading...", end=" ")
    if content == "$ERROR$":
        print("SKIP (error answer)")
    else:
        out = run_cmd(
            f"python gen_ground_truth_judgment.py --model {model} --question-source jsonl "
            f"--bench-name {bench} --question-end 1 --resume 2>&1"
        )
        if out is None:
            errors.append("Grading failed")
            print("FAIL")
        else:
            print("OK")

    # Clean up smoke test answer
    if os.path.exists(answer_file):
        os.remove(answer_file)

    # Result
    print()
    if errors:
        print(f"SMOKE TEST FAILED ({len(errors)} errors):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("SMOKE TEST PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
