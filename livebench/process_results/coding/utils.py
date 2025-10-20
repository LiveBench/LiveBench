import json
from pathlib import Path
from dataclasses import dataclass
import pickle
import zlib
import base64
from enum import Enum
import subprocess

from livebench.common import LIVE_BENCH_ROOT_PATH
from livebench.code_runner.eval import untrusted_check, PASS
from livebench.lcb_runner.utils.extraction_utils import extract_code
from livebench.lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics

import shutil

import shortuuid

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

def LCB_generation_process_results(question: dict, llm_answer: str, debug=False) -> int:

    extracted_answer = extract_code(model_output=llm_answer, lmstyle=None) # Missing out only on some slightly different handling for CodeLlamaInstruct from the original LiveCodeBench

    # if this is a completion question, check that the completion is present.
    if 'partial_solution' in question and (not question['partial_solution'] is None) and (len(question['partial_solution']) > 0) and not extracted_answer.startswith(question['partial_solution']):
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

def code_generation_process_results(question: dict, llm_answer: str, debug=False) -> int:
    extracted_code = extract_code(model_output=llm_answer, lmstyle=None)

    if 'partial_solution' in question and (not question['partial_solution'] is None) and (len(question['partial_solution']) > 0) and not extracted_code.startswith(question['partial_solution']):
        extracted_code = question['partial_solution'] + '\n' + extracted_code
    elif 'entry_point' in question and 'def ' + question['entry_point'] not in extracted_code:
        extracted_code = question['code_prompt'] + '\n' + extracted_code

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
    
# python agentic_code_runner/eval/harness/run_evaluation.py --config agentic_code_runner/eval/config.json
def agentic_coding_process_results(questions: list[dict], answers: list[dict], debug=False, max_workers=1, only_build_image=False) -> dict[str, int]:

    if len(answers) == 0:
        return dict()
    
    eval_id = shortuuid.uuid()

    config_path = Path(LIVE_BENCH_ROOT_PATH / 'agentic_code_runner/eval/config.json')
    config_path.parent.mkdir(parents=True, exist_ok=True)
    model_name = answers[0]['model_id']
    patch_path = Path(LIVE_BENCH_ROOT_PATH / f'agentic_code_runner/data/patches/{model_name}_{eval_id}_patch.jsonl')
    patch_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(LIVE_BENCH_ROOT_PATH / f'agentic_code_runner/data/dataset/{eval_id}.jsonl')
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    workdir_path = Path(LIVE_BENCH_ROOT_PATH / f'agentic_code_runner/data/workdir/{model_name}_{eval_id}')
    shutil.rmtree(workdir_path, ignore_errors=True)
    workdir_path.mkdir(parents=True, exist_ok=True)

    report_path = Path(LIVE_BENCH_ROOT_PATH / f'agentic_code_runner/data/report/{model_name}_{eval_id}/final_report.json')
    shutil.rmtree(report_path.parent, ignore_errors=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    log_path = Path(LIVE_BENCH_ROOT_PATH / f'agentic_code_runner/data/logs/{model_name}_{eval_id}/')
    shutil.rmtree(log_path, ignore_errors=True)
    log_path.mkdir(parents=True, exist_ok=True)

    repo_path = Path(LIVE_BENCH_ROOT_PATH / f'agentic_code_runner/data/repos')
    repo_path.mkdir(parents=True, exist_ok=True)

    with open(patch_path, 'w') as f:
        for question in questions:
            answer = [a for a in answers if a['question_id'] == question['question_id']]
            if len(answer) == 0:
                print(f"No answer found for question {question['question_id']}")
                continue
            answer = answer[0]
            answer_obj = {
                "org": question['org'],
                "repo": question['repo'],
                "number": question['number'],
                "fix_patch": answer['choices'][0]['turns'][0]
            }
            json.dump(answer_obj, f)
            f.write('\n')
    with open(dataset_path, 'w') as f:
        for question in questions:
            json.dump(question, f)
            f.write('\n')

    config = {
        "mode": "evaluation" if not only_build_image else "image",
        "workdir": f"{workdir_path.as_posix()}",
        "patch_files": [patch_path.as_posix()],
        "dataset_files": [dataset_path.as_posix()],
        "force_build": False,
        "output_dir": f"{report_path.parent.as_posix()}",
        "specifics": [],
        "skips": [],
        "repo_dir": f"{repo_path.as_posix()}",
        "need_clone": True,
        "global_env": [],
        "clear_env": True,
        "stop_on_error": False,
        "max_workers": max_workers,
        "max_workers_build_image": max_workers,
        "max_workers_run_instance": max_workers,
        "log_dir": f"{log_path.as_posix()}",
        "log_level": "DEBUG" if debug else "INFO"
    }
    with open(config_path, 'w') as f:
        json.dump(config, f)

    try:
        res = subprocess.run(['python', LIVE_BENCH_ROOT_PATH / 'agentic_code_runner/eval/harness/run_evaluation.py', '--config', config_path.as_posix()])
        if res.returncode != 0:
            print(f"Error running MSWEB evaluation: {res.returncode}")
            return dict()
    except Exception as e:
        print(f"Error running MSWEB evaluation: {e}")
        return dict()
    finally:
        config_path.unlink(missing_ok=True)
        patch_path.unlink(missing_ok=True)
        dataset_path.unlink(missing_ok=True)

    if only_build_image:
        return dict()

    report_path = Path(LIVE_BENCH_ROOT_PATH / f'agentic_code_runner/data/report/{model_name}_{eval_id}/final_report.json')
    if not report_path.exists():
        print(f"Report not found for eval {eval_id} for model {model_name}")
        return dict()
    report = json.load(open(report_path))

    result = {}

    for question in questions:
        question_id = question['question_id']
        instance_id = f"{question['org']}/{question['repo']}:pr-{question['number']}"
        if instance_id not in report['submitted_ids']:
            print(f"Instance {instance_id} not found in report (question {question_id})")
            result[question_id] = 0
        else:
            result[question_id] = 1 if instance_id in report['resolved_ids'] else 0
        
        if debug and result[question_id] == 0:
            if instance_id in report['unresolved_ids']:
                print(f"INCORRECT, {model_name} {question_id} ({instance_id})")
            elif instance_id in report['incomplete_ids']:
                print(f"INCOMPLETE, {model_name} {question_id} ({instance_id})")
            elif instance_id in report['empty_patch_ids']:
                print(f"EMPTY PATCH, {model_name} {question_id} ({instance_id})")
            elif instance_id in report['error_ids']:
                print(f"ERROR, {model_name} {question_id} ({instance_id})")
            print('RUN ID', answer['run_id'])
            print('EVAL ID', eval_id)
            print('WORKDIR', workdir_path)
    
    assert len(result) == len(questions)
    assert sum(result.values()) == report['resolved_instances']

    return result