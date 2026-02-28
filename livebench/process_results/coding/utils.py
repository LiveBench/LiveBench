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
from livebench.agentic_code_runner.eval.harness.constant import (
    EVALUATION_WORKDIR,
    FIX_PATCH_RUN_LOG_FILE,
    REPORT_FILE,
)
from livebench.agentic_code_runner.eval.harness.report import Report
from livebench.agentic_code_runner.eval.harness.test_result import TestStatus

import shutil

import shortuuid

# Number of times to retry failing agentic coding questions
AGENTIC_CODING_RETRIES = 2

# Maximum parallelism to use when retrying agentic coding questions
AGENTIC_CODING_RETRY_MAX_PARALLELISM = 3

# Patterns in fix-patch-run.log that indicate a retry-worthy issue
# Key: pattern name (for logging), Value: string to search for in log
AGENTIC_CODING_RETRY_PATTERNS = {
    "svelte_timeout": "Error: Test timed out in",
    "vue_flaky": "FAIL  packages/runtime-core/__tests__/components/Suspense.spec.ts > Suspense > mount the fallback content is in the correct position",
    "vue_other_flaky": "FAIL  packages/runtime-core/__tests__/components/Suspense.spec.ts > Suspense > branch switch to 3rd branch before resolve",
    "vue_other_other_flaky": "FAIL  packages/runtime-core/__tests__/components/Suspense.spec.ts > Suspense > nested suspense (w/ suspensible) switch several times before parent suspense resolve",
    "dayjs_badmutable": "FAIL test/plugin/badMutable.test.js"
}

# Question IDs that should be retried if incorrect (regardless of log patterns)
# These questions have flaky grading that cannot be identified by log patterns
AGENTIC_CODING_RETRY_QUESTIONS: list[str] = [
    "1af7e5326a1120764a6f82f8cde04187349a162e8a0d41f2b94b79a50a4e3cfa",
    "4f1cda742a90c6655c7c5de58c5bebf34508f394106d5c89aa90773bbb988e03",
    "1af7e5326a1120764a6f82f8cde04187349a162e8a0d41f2b94b79a50a4e3cfa"
]

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
    except Exception:
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


def _check_logs_for_retry_patterns(
    instance_map: dict[str, dict[str, str]], 
    question_ids: list[str], 
    retry_patterns: dict[str, str]
) -> dict[str, list[str]]:
    """
    Check fix-patch-run.log files for patterns indicating retry-worthy issues.
    
    Returns dict mapping question_id to list of matched pattern names.
    """
    retry_needed = {}
    
    for question_id in question_ids:
        if question_id not in instance_map:
            continue
            
        log_path = Path(instance_map[question_id]["fix_log_file"])
        if not log_path.exists():
            continue
            
        try:
            log_content = log_path.read_text()
            matched_patterns = []
            
            for pattern_name, pattern_string in retry_patterns.items():
                if pattern_string in log_content:
                    matched_patterns.append(pattern_name)
            
            if matched_patterns:
                retry_needed[question_id] = matched_patterns
        except Exception as e:
            print(f"Warning: Could not read log for {question_id}: {e}")
    
    return retry_needed


def _run_evaluation_subprocess(
    questions: list[dict],
    answers: list[dict],
    debug: bool,
    max_workers: int,
    only_build_image: bool = False
) -> tuple[dict[str, int], dict[str, dict[str, str]]]:
    """
    Run the evaluation subprocess and parse results.
    
    Returns:
        (results, instance_map) where results maps question_id -> score,
        and instance_map maps question_id -> metadata including log paths
    """
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
        "log_level": "DEBUG" if debug else "INFO",
        "instance_timeout": 480
    }
    with open(config_path, 'w') as f:
        json.dump(config, f)

    try:
        res = subprocess.run(['python', LIVE_BENCH_ROOT_PATH / 'agentic_code_runner/eval/harness/run_evaluation.py', '--config', config_path.as_posix()])
        if res.returncode != 0:
            print(f"Error running MSWEB evaluation: {res.returncode}")
            return dict(), dict()
    except Exception as e:
        print(f"Error running MSWEB evaluation: {e}")
        return dict(), dict()
    finally:
        config_path.unlink(missing_ok=True)
        patch_path.unlink(missing_ok=True)
        dataset_path.unlink(missing_ok=True)

    if only_build_image:
        return dict(), dict()

    report_path = Path(LIVE_BENCH_ROOT_PATH / f'agentic_code_runner/data/report/{model_name}_{eval_id}/final_report.json')
    if not report_path.exists():
        print(f"Report not found for eval {eval_id} for model {model_name}")
        return dict(), dict()
    report = json.load(open(report_path))

    result = {}
    instance_map: dict[str, dict[str, str]] = {}

    for question in questions:
        question_id = question['question_id']
        instance_id = f"{question['org']}/{question['repo']}:pr-{question['number']}"
        
        # Build instance_dir and paths for all questions (need for retry log checking)
        instance_dir: Path = (
            workdir_path
            / question['org']
            / question['repo']
            / EVALUATION_WORKDIR
            / f"pr-{question['number']}"
        )
        report_file = instance_dir / REPORT_FILE
        fix_log_file = instance_dir / FIX_PATCH_RUN_LOG_FILE

        instance_map[question_id] = {
            "instance_id": instance_id,
            "workdir": str(workdir_path),
            "instance_dir": str(instance_dir),
            "report_file": str(report_file),
            "fix_log_file": str(fix_log_file),
            "eval_id": eval_id,
            "model_name": model_name
        }
        
        if instance_id not in report['submitted_ids']:
            print(f"Instance {instance_id} not found in report (question {question_id})")
            result[question_id] = 0
        elif instance_id in report['error_ids'] or instance_id in report['incomplete_ids']:
            # Skip questions with infrastructure errors - don't include in result
            # This signals that grading should be re-run for these questions
            print(f"Skipping question {question_id} ({instance_id}) due to infrastructure error")
            continue
        else:
            result[question_id] = 1 if instance_id in report['resolved_ids'] else 0

        if debug and result.get(question_id) == 0:
            if instance_id in report['unresolved_ids']:
                print(f"INCORRECT, {model_name} {question_id} ({instance_id})")
            elif instance_id in report['incomplete_ids']:
                print(f"INCOMPLETE, {model_name} {question_id} ({instance_id})")
            elif instance_id in report['empty_patch_ids']:
                print(f"EMPTY PATCH, {model_name} {question_id} ({instance_id})")
            elif instance_id in report['error_ids']:
                print(f"ERROR, {model_name} {question_id} ({instance_id})")
            print('RUN ID', answers[[i for i, a in enumerate(answers) if a['question_id'] == question_id][0]]['run_id'])
            print('EVAL ID', eval_id)
            print('WORKDIR', workdir_path)

            def _failed_tests(report_obj: Report) -> list[tuple[str, TestStatus, TestStatus]]:
                failing: list[tuple[str, TestStatus, TestStatus]] = []
                for test_name, test_status in report_obj._tests.items():
                    if test_status.test == TestStatus.PASS and test_status.fix == TestStatus.FAIL:
                        failing.append((test_name, test_status.test, test_status.fix))
                    elif test_status.test == TestStatus.FAIL and test_status.fix != TestStatus.PASS:
                        failing.append((test_name, test_status.test, test_status.fix))
                return failing

            def _format_status(test_status: TestStatus) -> str:
                return test_status.value if isinstance(test_status, TestStatus) else str(test_status)

            if report_file.exists():
                try:
                    report_obj = Report.from_json(report_file.read_text())
                    failing_tests = _failed_tests(report_obj)
                    if failing_tests:
                        print('Failing tests (test -> fix):')
                        for name, test_status, fix_status in failing_tests:
                            print(f'  {name}: { _format_status(test_status)} -> { _format_status(fix_status)}')
                    else:
                        print('No failing tests identified in report.')
                except Exception as e:
                    print(f'Could not parse report for {instance_id}: {e}')
            else:
                print(f'Report file not found at {report_file}')

            if fix_log_file.exists():
                print(f'Fix patch log: {fix_log_file}')
            else:
                print(f'Fix patch log missing: {fix_log_file}')
    
    # Note: result may have fewer entries than questions if some had infrastructure errors
    assert len(result) <= len(questions)
    assert sum(result.values()) == report['resolved_instances']

    mapping_path = report_path.parent / "question_instance_map.json"
    with open(mapping_path, 'w') as f:
        json.dump(instance_map, f, indent=2)

    return result, instance_map


# python agentic_code_runner/eval/harness/run_evaluation.py --config agentic_code_runner/eval/config.json
def agentic_coding_process_results(questions: list[dict], answers: list[dict], debug=False, max_workers=1, only_build_image=False) -> dict[str, int]:

    if len(answers) == 0:
        return dict()
    
    # Initial evaluation
    result, instance_map = _run_evaluation_subprocess(questions, answers, debug, max_workers, only_build_image)
    
    # Retry logic
    if AGENTIC_CODING_RETRIES > 0 and not only_build_image:
        # Track questions from retry list that have already been retried once
        retry_list_already_retried: set[str] = set()
        
        for retry_num in range(AGENTIC_CODING_RETRIES):
            # Get failing question IDs (including error/incomplete that aren't in result)
            all_question_ids = [q['question_id'] for q in questions]
            failing_question_ids = [qid for qid in all_question_ids if result.get(qid, 0) == 0]
            
            if not failing_question_ids:
                break
            
            # Check logs for retry-worthy patterns
            retry_needed = _check_logs_for_retry_patterns(
                instance_map, 
                failing_question_ids, 
                AGENTIC_CODING_RETRY_PATTERNS
            )
            
            # Add questions from AGENTIC_CODING_RETRY_QUESTIONS that are failing (only if not already retried)
            for qid in failing_question_ids:
                if qid in AGENTIC_CODING_RETRY_QUESTIONS and qid not in retry_list_already_retried:
                    if qid in retry_needed:
                        retry_needed[qid].append("in_retry_questions_list")
                    else:
                        retry_needed[qid] = ["in_retry_questions_list"]
                    # Mark this question as having been retried from the list
                    retry_list_already_retried.add(qid)
            
            # Add questions with errors/incomplete (not in result) - these should always be retried
            for qid in all_question_ids:
                if qid not in result:
                    if qid in retry_needed:
                        retry_needed[qid].append("error_or_incomplete")
                    else:
                        retry_needed[qid] = ["error_or_incomplete"]
            
            if not retry_needed:
                print(f"No retry-worthy patterns found in logs, skipping remaining retries")
                break
            
            # Filter to questions that need retry
            retry_questions = [q for q in questions if q['question_id'] in retry_needed]
            retry_answers = [a for a in answers if a['question_id'] in retry_needed]
            
            print(f"Retry {retry_num + 1}/{AGENTIC_CODING_RETRIES}: Found {len(retry_questions)} questions with retry-worthy issues")
            for qid, patterns in retry_needed.items():
                print(f"  {qid}: {', '.join(patterns)}")
            
            # Retry with reduced parallelism
            retry_max_workers = min(max_workers, AGENTIC_CODING_RETRY_MAX_PARALLELISM)
            print(f"  Using max_workers={retry_max_workers} (reduced from {max_workers})")
            
            # Run retry evaluation and get UPDATED instance_map
            retry_result, retry_instance_map = _run_evaluation_subprocess(
                retry_questions, 
                retry_answers, 
                debug, 
                retry_max_workers,
                only_build_image
            )
            
            # Update results with any successes
            for question_id, score in retry_result.items():
                if score == 1:
                    result[question_id] = 1
            
            # Update instance_map with new paths for next iteration
            instance_map.update(retry_instance_map)
    
    return result