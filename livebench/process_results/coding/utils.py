import json
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import pickle
import copy
import zlib
import base64
import re
from enum import Enum

from livebench.bcb_runner.eval import untrusted_check, PASS, FAIL, TIMEOUT
from livebench.lcb_runner.utils.extraction_utils import extract_code
from livebench.lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics

# from LiveCodebench, modified

class TestType(Enum):
    STDIN = "stdin"
    FUNCTIONAL = "functional"

@dataclass
class Test:
    input: str
    output: str
    testtype: TestType

    def __post_init__(self):
        self.testtype = TestType(self.testtype)
        # if self.testtype == TestType.FUNCTIONAL:
        #     self.input = json.loads(self.input)
        #     self.output = json.loads(self.output)

def LCB_generation_process_results(question: dict, llm_answer: str, debug=False) -> int:

    extracted_answer = extract_code(model_output=llm_answer, lmstyle=None) # Missing out only on some slightly different handling for CodeLlamaInstruct from the original LiveCodeBench

    # if this is a completion question, check that the completion is present.
    if 'partial_solution' in question and (not question['partial_solution'] is None) and (len(question['partial_solution']) > 0) and not extracted_answer.startswith(question['partial_solution']):
        # if len(llm_answer) < len(question['partial_solution']):
        #     return 0
        # if llm_answer[:len(question['partial_solution'])] != question['partial_solution']:
        #     return 0
        full_solution = question['partial_solution'] + '\n' + extracted_answer
    else:
        full_solution = extracted_answer

    # code mostly from LiveCodeBench, with modifications.
    public_test_cases = json.loads(question['public_test_cases'])  # type: ignore
    public_test_cases = [Test(**t) for t in public_test_cases]

    try:
        private_test_cases = json.loads(question['private_test_cases'])  # type: ignore
    except:
        private_test_cases = json.loads(
            pickle.loads(
                zlib.decompress(
                    base64.b64decode(question['private_test_cases'].encode('utf-8'))  # type: ignore
                )
            )
        )  # type: ignore

    private_test_cases = [Test(**t) for t in private_test_cases]
    metadata = json.loads(question['original_json']['metadata'])  # type: ignore
    eval_sample = {
        "input_output": json.dumps(
            {
                "inputs": [
                    t.input
                    for t in public_test_cases + private_test_cases
                ],
                "outputs": [
                    t.output
                    for t in public_test_cases + private_test_cases
                ],
                "fn_name": metadata.get("func_name", None),
            }
        )
    }

    metrics, results, metadata = codegen_metrics(
        [eval_sample],
        [[full_solution]],
        k_list=[1], # can't compute higher pass@ because we don't have more than one prediction.
        num_process_evaluate=1, # multiprocessing is handled at a higher level to parallelize multiple questions at once, so we don't want to complicate this with forking here.
        timeout=6, # default eval setting from livecodebench.
    )

    if metrics['pass@1'] == 1.0:
        return 1
    else:
        if debug:
            print('INCORRECT', question['question_title'], question['question_id'])
            if 'partial_solution' in question and (not question['partial_solution'] is None) and (len(question['partial_solution']) > 0) and not extracted_answer.startswith(question['partial_solution']):
                print('starter code')
                print(question['partial_solution'])
            print('extracted answer')
            print(extracted_answer)
            if extracted_answer != llm_answer.replace('\n```', '').rstrip():
                print('original llm answer')
                print(llm_answer)
            print('results', results)
            print('metadata', metadata)
        return 0

def BCB_generation_process_results(question: dict, llm_answer: str, debug=False) -> int:
    extracted_code = extract_code(model_output=llm_answer, lmstyle=None)

    if 'partial_solution' in question and (not question['partial_solution'] is None) and (len(question['partial_solution']) > 0) and not extracted_code.startswith(question['partial_solution']):
        extracted_code = question['partial_solution'] + '\n' + extracted_code

    test_cases = question['tests']
    expected_time = question['expected_time']

    stat, details = untrusted_check( # defaults from bigcodebench
        code=extracted_code,
        test_code=test_cases,
        entry_point=question['entry_point'],
        max_as_limit=30 * 1024,
        max_data_limit=30 * 1024,
        max_stack_limit=10,
        min_time_limit=1,
        gt_time_limit=expected_time + 3 if 'expected_time' in question else 20
    )

    if stat == PASS:
        return 1
    else:
        if debug:
            print('INCORRECT', question['question_title'], question['question_id'])
            print('extracted code')
            print(extracted_code)
            print('stat', stat)
            print('details:')
            for test_id in details:
                if test_id == '_captured_stdout_' or test_id == '_captured_stderr_' or test_id == '_exception_':
                    continue
                description, trace = details[test_id]
                print(f'  Test {test_id}:')
                print(f'    Description: {description}')
                print(f'    Traceback: {trace}')
                print()
            if '_captured_stdout_' in details and details['_captured_stdout_'].strip() != '':
                print('captured stdout:')
                print(details['_captured_stdout_'])
            if '_captured_stderr_' in details and details['_captured_stderr_'].strip() != '':
                print('captured stderr:')
                print(details['_captured_stderr_'])
            if '_exception_' in details and details['_exception_'].strip() != '':
                print('exception:')
                print(details['_exception_'])
                
        return 0
    
    
