"""Total API spend per task and per run from model_answer/<model>.jsonl files.

Uses the shared methodology in scripts/cost_utils.py (same rules as
generate_model_rows.py): cost from token counts x config pricing with
per-answer detection of the two input-token conventions, cache-write
billing for Anthropic (excluded on runaway step-cap answers), one
(latest) answer counted per question, and $ERROR$/-1 sentinels excluded
from token averages. Provider cost_usd is only used when the config has
no pricing.

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cost_utils import active_qids, answer_cost, is_anthropic, latest_per_question, load_price

from livebench.common import LIVE_BENCH_ROOT_PATH


def main():
    parser = argparse.ArgumentParser(description="Total API spend for a model's run")
    parser.add_argument('--model', required=True, help='model display name (e.g. minimax-m2.7)')
    parser.add_argument('--bench-name', default='live_bench',
                        help='subtree to scan, e.g. live_bench or live_bench/math (default: live_bench)')
    parser.add_argument('--active-only', action='store_true',
                        help='count only questions with empty livebench_removal_date')
    args = parser.parse_args()

    price = load_price(args.model)
    charge_cache_write = is_anthropic(args.model)
    in_price = out_price = None
    if price:
        in_price, cached_price, out_price, cc_price = price

    model_file = f'{args.model.lower()}.jsonl'
    pattern = str(LIVE_BENCH_ROOT_PATH / 'data' / args.bench_name / '**' / 'model_answer' / model_file)
    answer_files = sorted(glob.glob(pattern, recursive=True))
    if not answer_files:
        print(f'No answer files for {args.model} under {args.bench_name}')
        return

    grand = {'cost': 0.0, 'in_tok': 0, 'out_tok': 0, 'n': 0, 'no_tok': 0}
    rows = []
    cats = {}
    for af in answer_files:
        task = af.split('/data/')[-1].rsplit('/model_answer/', 1)[0]
        # task path is live_bench/<category>/<task>
        category = task.split('/')[1] if '/' in task else task
        keep = active_qids(os.path.dirname(os.path.dirname(af))) if args.active_only else None
        records = (json.loads(l) for l in open(af) if l.strip())
        answers = latest_per_question(records, keep)
        in_tok = out_tok = no_tok = 0
        valid_n = valid_out = 0
        cost = 0.0
        for a in answers.values():
            it = max(0, a.get('total_input_tokens') or 0)
            ot = max(0, a.get('total_output_tokens') or 0)
            in_tok += it
            out_tok += ot
            if price is not None:
                cost += answer_cost(a, in_price, cached_price, out_price, cc_price, charge_cache_write)
            else:
                cost += a.get('cost_usd') or 0.0
            if ot > 0:
                valid_n += 1
                valid_out += ot
            else:
                no_tok += 1
        n = len(answers)
        if n == 0:
            continue
        rows.append((task, n, in_tok, out_tok, cost, no_tok))
        grand['cost'] += cost
        grand['in_tok'] += in_tok
        grand['out_tok'] += out_tok
        grand['n'] += n
        grand['no_tok'] += no_tok
        c = cats.setdefault(category, {'n': 0, 'valid_n': 0, 'valid_out': 0, 'cost': 0.0})
        c['n'] += n
        c['valid_n'] += valid_n
        c['valid_out'] += valid_out
        c['cost'] += cost

    rows.sort(key=lambda x: -x[4])
    print(f"\nSpend report for {args.model}"
          + (f"  (price: ${in_price}/1M in, ${out_price}/1M out)" if price else "  (no price in config; using provider cost_usd)"))
    print(f"{'task':<42}{'n':>5}{'in_tok':>12}{'out_tok':>12}{'cost_usd':>12}")
    print('-' * 83)
    for task, n, it, ot, cost, no_tok in rows:
        flag = ' *' if no_tok else ''
        print(f"{task:<42}{n:>5}{it:>12,}{ot:>12,}{cost:>12.4f}{flag}")
    print('-' * 83)
    print(f"{'TOTAL':<42}{grand['n']:>5}{grand['in_tok']:>12,}{grand['out_tok']:>12,}{grand['cost']:>12.4f}")

    print("\nPer-category averages (over answers with token data; cost = tokens x config price):")
    print(f"{'category':<26}{'n':>5}{'avg_out_tok/q':>15}{'avg_cost/q':>12}{'total_cost':>12}")
    print('-' * 70)
    for category in sorted(cats):
        c = cats[category]
        avg_out = c['valid_out'] / c['valid_n'] if c['valid_n'] else 0
        avg_cost = c['cost'] / c['n'] if c['n'] else 0
        print(f"{category:<26}{c['n']:>5}{avg_out:>15,.0f}{avg_cost:>12.4f}{c['cost']:>12.4f}")

    if grand['no_tok']:
        print(f"\n* {grand['no_tok']} answers had no token data ($ERROR$ or predate cost tracking) -> "
              f"cost 0 for those; totals are a lower bound.")
    if price is None:
        print("\nNo cost_per_million in the model config; add it to compute spend from tokens.")


if __name__ == '__main__':
    main()
