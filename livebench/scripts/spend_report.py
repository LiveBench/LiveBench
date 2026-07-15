"""Total API spend per task and per run from model_answer/<model>.jsonl files.

Cost is computed from token counts x config pricing (same methodology as
scripts/generate_model_rows.py — the provider-reported cost_usd is
inconsistent across providers and is only used when the config has no
pricing): uncached_in*input + cache_read*cached_input + output*output,
plus cache_creation*cache_creation price for models that report cache
writes (Anthropic).

Usage:
    python scripts/spend_report.py --model minimax-m2.7
    python scripts/spend_report.py --model minimax-m2.7 --bench-name live_bench/math
    python scripts/spend_report.py --model minimax-m2.7 --active-only
"""
import argparse
import glob
import json
import os

from livebench.common import LIVE_BENCH_ROOT_PATH
from livebench.model import get_model_config


def load_price(model: str):
    """Return (input, cached_input, output, cache_creation) USD per 1M tokens, or None.
    Missing cached / cache_creation default to the standard 0.1x / 1.25x of input."""
    try:
        cfg = get_model_config(model)
        cpm = getattr(cfg, 'cost_per_million', None)
    except Exception:
        cpm = None
    if not cpm:
        return None
    in_p = float(cpm.get('input', 0))
    return (in_p, float(cpm.get('cached_input', in_p * 0.1)),
            float(cpm.get('output', 0)), float(cpm.get('cache_creation', in_p * 1.25)))


def main():
    parser = argparse.ArgumentParser(description="Total API spend for a model's run")
    parser.add_argument('--model', required=True, help='model display name (e.g. minimax-m2.7)')
    parser.add_argument('--bench-name', default='live_bench',
                        help='subtree to scan, e.g. live_bench or live_bench/math (default: live_bench)')
    parser.add_argument('--active-only', action='store_true',
                        help='count only questions with empty livebench_removal_date')
    args = parser.parse_args()

    price = load_price(args.model)
    in_price, cached_price, out_price, cc_price = price if price else (None, None, None, None)

    model_file = f'{args.model.lower()}.jsonl'
    pattern = str(LIVE_BENCH_ROOT_PATH / 'data' / args.bench_name / '**' / 'model_answer' / model_file)
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
    cats = {}
    for af in answer_files:
        task = af.split('/data/')[-1].rsplit('/model_answer/', 1)[0]
        # task path is live_bench/<category>/<task>
        category = task.split('/')[1] if '/' in task else task
        keep = active_ids(af) if args.active_only else None
        in_tok = out_tok = n = partial = 0
        valid_n = valid_out = 0
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
            cc = r.get('total_cache_creation_tokens') or 0
            ot = r.get('total_output_tokens') or 0
            ot = ot if ot > 0 else 0
            in_tok += it
            out_tok += ot
            # cost from tokens x config price (generate_model_rows methodology);
            # provider cost_usd only when the config has no pricing
            if price is not None:
                cached = min(ct, it)
                uncached = max(it - cached, 0)
                c = ((uncached / 1_000_000) * in_price + (cached / 1_000_000) * cached_price
                     + (ot / 1_000_000) * out_price + (cc / 1_000_000) * cc_price)
                if not it and not ot:
                    partial += 1
            else:
                c = r.get('cost_usd') or 0.0
            cost += c
            # averages basis: answers with real token data (excludes $ERROR$/empty)
            if ot > 0:
                valid_n += 1
                valid_out += ot
        if n == 0:
            continue
        rows.append((task, n, in_tok, out_tok, cost, partial))
        grand['cost'] += cost
        grand['in_tok'] += in_tok
        grand['out_tok'] += out_tok
        grand['n'] += n
        grand['partial'] += partial
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
    for task, n, it, ot, cost, partial in rows:
        flag = ' *' if partial else ''
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

    if grand['partial']:
        print(f"\n* {grand['partial']} answers had no token data (predate cost tracking or $ERROR$) -> "
              f"their cost is counted as 0; total is a lower bound.")
    if price is None:
        print("\nNo cost_per_million in the model config; add it to compute spend from tokens.")


if __name__ == '__main__':
    main()
