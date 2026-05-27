#!/usr/bin/env python3
"""Total API spend per task and per run from model_answer/<model>.jsonl files.

Usage:
    python scripts/spend_report.py --model minimax-m2.7
    python scripts/spend_report.py --model minimax-m2.7 --bench-name live_bench/math
    python scripts/spend_report.py --model minimax-m2.7 --active-only
"""
import argparse
import glob
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from livebench.model.api_model_config import get_model_config

DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', 'data')


def load_price(model: str):
    """Return (input, cached_input, output) USD per 1M tokens, or None."""
    try:
        cfg = get_model_config(model)
        cpm = getattr(cfg, 'cost_per_million', None)
    except Exception:
        cpm = None
    if not cpm:
        return None
    in_p = float(cpm.get('input', 0))
    return in_p, float(cpm.get('cached_input', in_p)), float(cpm.get('output', 0))


def main():
    ap = argparse.ArgumentParser(description="Total API spend for a model's run")
    ap.add_argument('--model', required=True, help='model display name (e.g. minimax-m2.7)')
    ap.add_argument('--bench-name', default='live_bench',
                    help='subtree to scan, e.g. live_bench or live_bench/math (default: live_bench)')
    ap.add_argument('--active-only', action='store_true',
                    help='count only questions with empty livebench_removal_date')
    args = ap.parse_args()

    price = load_price(args.model)
    in_price, cached_price, out_price = price if price else (None, None, None)

    model_file = f'{args.model.lower()}.jsonl'
    pattern = os.path.join(DATA_ROOT, args.bench_name, '**', 'model_answer', model_file)
    answer_files = sorted(glob.glob(pattern, recursive=True))
    if not answer_files:
        print(f'No answer files for {args.model} under {args.bench_name}')
        return

    def active_ids(answer_file):
        qf = os.path.join(os.path.dirname(os.path.dirname(answer_file)), 'question.jsonl')
        ids = set()
        if os.path.exists(qf):
            for line in open(qf):
                if line.strip():
                    q = json.loads(line)
                    if q.get('livebench_removal_date', '') == '':
                        ids.add(q['question_id'])
        return ids

    grand = {'cost': 0.0, 'in_tok': 0, 'out_tok': 0, 'n': 0, 'partial': 0}
    rows = []
    for af in answer_files:
        task = af.split('/data/')[-1].rsplit('/model_answer/', 1)[0]
        keep = active_ids(af) if args.active_only else None
        in_tok = out_tok = n = partial = 0
        cost = 0.0
        for line in open(af):
            if not line.strip():
                continue
            r = json.loads(line)
            if keep is not None and r.get('question_id') not in keep:
                continue
            n += 1
            it = r.get('total_input_tokens') or 0
            ct = r.get('total_cached_tokens') or 0
            ot = r.get('total_output_tokens') or 0
            ot = ot if ot > 0 else 0
            in_tok += it
            out_tok += ot
            c = r.get('cost_usd')
            if c is None and price is not None:
                cached = min(ct, it)
                uncached = max(it - cached, 0)
                c = (uncached / 1e6) * in_price + (cached / 1e6) * cached_price + (ot / 1e6) * out_price
                if not it:
                    partial += 1
            cost += (c or 0.0)
        if n == 0:
            continue
        rows.append((task, n, in_tok, out_tok, cost, partial))
        grand['cost'] += cost
        grand['in_tok'] += in_tok
        grand['out_tok'] += out_tok
        grand['n'] += n
        grand['partial'] += partial

    rows.sort(key=lambda x: -x[4])
    print(f"\nSpend report for {args.model}"
          + (f"  (price: ${in_price}/1M in, ${out_price}/1M out)" if price else "  (no price in config)"))
    print(f"{'task':<42}{'n':>5}{'in_tok':>12}{'out_tok':>12}{'cost_usd':>12}")
    print('-' * 83)
    for task, n, it, ot, cost, partial in rows:
        flag = ' *' if partial else ''
        print(f"{task:<42}{n:>5}{it:>12,}{ot:>12,}{cost:>12.4f}{flag}")
    print('-' * 83)
    print(f"{'TOTAL':<42}{grand['n']:>5}{grand['in_tok']:>12,}{grand['out_tok']:>12,}{grand['cost']:>12.4f}")
    if grand['partial']:
        print(f"\n* {grand['partial']} answers had no input-token data (predate cost tracking) -> "
              f"cost is OUTPUT-ONLY for those; total is a lower bound.")
    if price is None:
        print("\nNo cost_per_million in the model config; add it to compute spend.")


if __name__ == '__main__':
    main()
