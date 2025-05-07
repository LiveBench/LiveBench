import re
import numpy as np
from ast import literal_eval
import json
import ast
from livebench.process_results.util import last_boxed_only_string, remove_boxed
from livebench.types import TablejoinQuestion

def clean_llm_output(s):
    pattern_solution = r'<solution>(.*?)</solution>'
    matches = re.findall(pattern_solution, s, re.DOTALL)
    if len(matches) > 0:
        return clean_llm_output(matches[-1].strip())
    try:
        match_d = literal_eval(s)
    except:
        matches = re.findall('%s(.*?)%s' % ("```python", "```"), s.replace("\n",""),re.MULTILINE)
        if len(matches) == 0:
            matches = re.findall('%s(.*?)%s' % ("```json", "```"), s.replace("\n",""),re.MULTILINE)
        if len(matches) == 0:
            matches = re.findall('%s(.*?)%s' % ("```", "```"), s.replace("\n",""),re.MULTILINE)
        if len(matches) == 0:
            if '\\boxed' in s:
                boxed = last_boxed_only_string(s.replace('\n', ''))
                if boxed:
                    no_boxed = remove_boxed(boxed)
                    matches = [re.sub(r"\\text{[\'|\"](.*?)[\'|\"]}", r"'\1'", no_boxed).replace('\\', '')]
        if len(matches) == 0:
            matches = [s]
        if len(matches) >= 1:
            matches = matches[-1]
        matches = matches.replace('null', 'None')
        try:
            match_d = literal_eval(matches)
        except:
            return {}
    if not isinstance(match_d, dict):
        return {}
    else:
        keys = list(match_d.keys())
        for k in keys:
            if match_d[k] is None:
                del match_d[k]
        return match_d

def joinmap_process_results(question: TablejoinQuestion, llm_answer: str, debug: bool =False):
    if not isinstance(question.ground_truth, dict):
        question.ground_truth = ast.literal_eval(question.ground_truth)
    llm_clean = clean_llm_output(llm_answer)
    if len(llm_clean) == 0:
        if debug:
            print('could not parse output')
            print('GROUND TRUTH', question.ground_truth)
            print('END OF OUTPUT', llm_answer[-min(500, len(llm_answer)):])
        return 0.0
    tp = 0
    fp = 0
    fn = 0
    for k, v in llm_clean.items():
        gt = question.ground_truth.get(k, None)
        if not gt:
            fp += 1
        elif gt == v:
            tp += 1
        else:
            fp += 1
            fn += 1
    if isinstance(question.ground_truth, str):
        question.ground_truth = json.loads(question.ground_truth)
    for k, v in question.ground_truth.items():
        llm_resp = llm_clean.get(k, None)
        if not llm_resp:
            fn += 1
    result = np.round(((2 * tp) / ((2 * tp) + fp + fn)), 2)
    if debug and result < 1:
        print('INCORRECT')
        print('GROUND TRUTH', question.ground_truth)
        print('SOLUTION', llm_clean)
        print('END OF OUTPUT', llm_answer[-min(500, len(llm_answer)):])
    return result 
