"""Shared cost/token accounting for eval-data scripts.

Single source of truth for the methodology used by spend_report.py here and
generate_model_rows.py in livebench-private / new-livebench (which vendors the
same logic to stay standalone). If you change the rules here, mirror them there.

Rules:
  - cost = uncached_input*input + cache_read*cached_input + output*output, plus
    cache_creation*cache_creation price for Anthropic models — except on runaway
    answers at the RUNAWAY_CALLS step cap, whose cache-writes are harness
    artifacts (the thrashing agent re-wrote the context every call).
  - total_input_tokens follows two conventions, auto-detected per answer:
    Anthropic-native reports it EXCLUSIVE of cache reads (cache_read can exceed
    input); OpenAI/xAI (and some Anthropic runs) report it INCLUSIVE.
  - $ERROR$ / -1 token sentinels clamp to 0 and are excluded from token averages.
  - The provider-reported cost_usd is NOT used when config pricing exists: it is
    inconsistent across providers (undiscounted cache reads, $0 records).
"""
import json
import os

RUNAWAY_CALLS = 250  # agent step cap; see cache-write note above


def load_price(model: str):
    """(input, cached_input, output, cache_creation) USD/1M from the config, or None.
    Missing cached / cache_creation default to the standard 0.1x / 1.25x of input."""
    try:
        from livebench.model import get_model_config
        cpm = getattr(get_model_config(model), 'cost_per_million', None)
    except Exception:
        cpm = None
    if not cpm:
        return None
    in_p = float(cpm.get('input', 0))
    return (in_p, float(cpm.get('cached_input', in_p * 0.1)),
            float(cpm.get('output', 0)), float(cpm.get('cache_creation', in_p * 1.25)))


def is_anthropic(model: str) -> bool:
    try:
        from livebench.model import get_model_config
        an = getattr(get_model_config(model), 'api_name', {})
        return 'anthropic' in (an if isinstance(an, dict) else {})
    except Exception:
        return model.startswith('claude')


def answer_cost(a: dict, in_price, cached_price, out_price, cache_write_price, charge_cache_write) -> float:
    """Per-answer cost from token counts x config pricing (uniform across providers)."""
    ti = max(0, a.get('total_input_tokens', 0) or 0)   # clamp $ERROR$/-1 sentinels to 0
    to = max(0, a.get('total_output_tokens', 0) or 0)
    cr = max(0, a.get('total_cached_tokens', 0) or 0)
    cc = max(0, a.get('total_cache_creation_tokens', 0) or 0)
    ncalls = a.get('n_model_calls') or 0
    # EXCLUSIVE convention (cr > ti): input is uncached only. INCLUSIVE: input
    # already contains cache_read (+ cache_creation), so subtract both.
    uncached = ti if cr > ti else max(0, ti - cr - cc)
    cost = (uncached * in_price + cr * cached_price + to * out_price) / 1e6
    if charge_cache_write and cc and ncalls < RUNAWAY_CALLS:
        cost += cc * cache_write_price / 1e6
    return cost


def active_qids(task_dir: str) -> set:
    """Question ids that are currently live for a task (empty livebench_removal_date)."""
    qf = os.path.join(task_dir, 'question.jsonl')
    ids = set()
    if os.path.exists(qf):
        for line in open(qf):
            if line.strip():
                q = json.loads(line)
                if q.get('livebench_removal_date', '') == '':
                    ids.add(q['question_id'])
    return ids


def latest_per_question(records, keep: set | None = None) -> dict:
    """One record per question_id — the latest by tstamp. Answer files accumulate
    re-runs; counting every record over-counts spend. Optionally restrict to `keep`."""
    latest = {}
    for r in records:
        qid = r.get('question_id')
        if keep is not None and qid not in keep:
            continue
        cur = latest.get(qid)
        if cur is None or (r.get('tstamp') or 0) >= (cur.get('tstamp') or 0):
            latest[qid] = r
    return latest
