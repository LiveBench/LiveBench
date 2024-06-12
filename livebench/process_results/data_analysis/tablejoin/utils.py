import re
import numpy as np
from ast import literal_eval
import json

def clean_llm_output(s):
    matches = re.findall('%s(.*)%s' % ("```python", "```"), s.replace("\n",""),re.MULTILINE)
    if len(matches) == 0:
        matches = [s]
    try:
        match_d = literal_eval(matches[-1])
    except:
        return {}
    if not isinstance(match_d, dict):
        return {}
    else:
        return match_d

def joinmap_process_results(_, ground_truth, llm):
    llm_clean = clean_llm_output(llm)
    if len(llm_clean) == 0:
        return 0.0
    tp = 0
    fp = 0
    fn = 0
    for k, v in llm_clean.items():
        gt = ground_truth.get(k, None)
        if not gt:
            fp += 1
        elif gt == v:
            tp += 1
        else:
            fp += 1
            fn += 1
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)
    for k, v in ground_truth.items():
        llm_resp = llm_clean.get(k, None)
        if not llm_resp:
            fn += 1
    result = np.round(((2 * tp) / ((2 * tp) + fp + fn)), 2)
    return result 