import argparse
import json
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from livebench.common import LIVE_BENCH_ROOT_PATH
from livebench.agentic_code_runner.eval.harness.report import Report
from livebench.agentic_code_runner.eval.harness.test_result import TestStatus
from livebench.process_results.coding.utils import agentic_coding_process_results

parser = argparse.ArgumentParser(description='Check the grading flakiness of a task for a given model\'s answers')
parser.add_argument('--model', required=True, nargs='+', help='Name of the model(s) to check')
parser.add_argument('--bench-name', required=True, help='Name of the benchmark to check')
parser.add_argument('--question-id', required=False, nargs='+', help='Question IDs to check')
parser.add_argument('--question-source', required=False, help='Question source')
parser.add_argument('--parallel', required=False, type=int, help='Number of parallel grading threads')

args = parser.parse_args()

num_iterations = 5

bench_name = args.bench_name
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path("flakiness_testing") / bench_name / timestamp
output_dir.mkdir(parents=True, exist_ok=True)
temp_files = [(output_dir / f"iteration_{i+1}.jsonl").as_posix() for i in range(num_iterations)]

is_agentic = "agentic_coding" in bench_name
report_root = Path(LIVE_BENCH_ROOT_PATH / 'agentic_code_runner/data/report')

def snapshot_runs() -> set[str]:
    if not report_root.exists():
        return set()
    return {p.name for p in report_root.iterdir() if p.is_dir()}

def load_agentic_artifacts(run_dir: Path) -> dict[tuple[str, str], list[dict]]:
    artifacts: dict[tuple[str, str], list[dict]] = defaultdict(list)
    mapping_file = run_dir / "question_instance_map.json"
    if not mapping_file.exists():
        return artifacts
    try:
        mapping = json.loads(mapping_file.read_text())
    except Exception:
        return artifacts

    for question_id, info in mapping.items():
        key = (question_id, info.get("model_name", ""))
        artifacts[key].append(info)
    return artifacts

def extract_failing_tests(report_path: Path) -> list[tuple[str, str, str]]:
    try:
        report_obj = Report.from_json(report_path.read_text())
    except Exception:
        return []
    failures: list[tuple[str, str, str]] = []
    for name, test_status in report_obj._tests.items():
        test_phase = test_status.test.value if isinstance(test_status.test, TestStatus) else str(test_status.test)
        fix_phase = test_status.fix.value if isinstance(test_status.fix, TestStatus) else str(test_status.fix)
        if test_status.test == TestStatus.PASS and test_status.fix == TestStatus.FAIL:
            failures.append((name, test_phase, fix_phase))
        elif test_status.test == TestStatus.FAIL and test_status.fix != TestStatus.PASS:
            failures.append((name, test_phase, fix_phase))
    return failures

def load_questions(bench_name: str, question_ids: list[str] | None = None, question_source: str | None = None) -> list[dict]:
    """Load questions from the benchmark question file."""
    question_file = Path("data") / bench_name / "question.jsonl"
    
    if not question_file.exists():
        raise FileNotFoundError(f"Question file not found: {question_file}")
    
    questions = []
    with open(question_file, 'r') as f:
        for line in f:
            question = json.loads(line)
            # Filter by question_id if specified
            if question_ids and question['question_id'] not in question_ids:
                continue
            questions.append(question)
    
    print(f"Loaded {len(questions)} questions from {question_file}")
    return questions

def grade_ground_truth(bench_name: str, question_ids: list[str] | None, question_source: str | None, parallel: int | None) -> dict[str, float]:
    """Grade ground truth answers for agentic coding questions."""
    questions = load_questions(bench_name, question_ids, question_source)
    
    if not questions:
        print("Warning: No questions loaded!")
        return {}
    
    # Create ground truth answers
    gt_answers = []
    for question in questions:
        answer = {
            'model_id': 'ground_truth',
            'choices': [{'turns': [question['fix_patch']]}],
            'question_id': question['question_id'],
            'answer_id': 'placeholder',
            'run_id': 'placeholder'
        }
        gt_answers.append(answer)
    
    print(f"Running evaluation on {len(questions)} questions with {parallel or 1} workers...")
    
    # Process results and get scores
    max_workers = parallel if parallel else 1
    result = agentic_coding_process_results(questions, gt_answers, debug=True, max_workers=max_workers)
    
    print(f"Evaluation complete. Results: {len(result)} question(s)")
    
    # Convert boolean results to scores (1.0 for success, 0.0 for failure)
    scores = {question_id: 1.0 if success else 0.0 for question_id, success in result.items()}
    return scores

agentic_artifacts: dict[tuple[str, str], list[dict]] = defaultdict(list)

# Separate ground_truth model from regular models
regular_models = [m for m in args.model if m != "ground_truth"]
has_ground_truth = "ground_truth" in args.model

# Run gen_ground_truth_judgment.py 5 times
existing_runs = snapshot_runs() if is_agentic else set()
for i, temp_file in enumerate(temp_files):
    print(f"Running grading iteration {i+1}/{num_iterations}...")
    
    # Handle regular models
    if regular_models:
        cmd = [
            'python', 'gen_ground_truth_judgment.py',
            '--bench-name', bench_name,
            '--model', *regular_models,
            '--output-file', temp_file,
        ]
        
        if args.question_id:
            cmd.extend(['--question-id', *args.question_id])
        
        if args.question_source:
            cmd.extend(['--question-source', args.question_source])
        
        if args.parallel:
            cmd.extend(['--parallel', str(args.parallel)])
        
        subprocess.run(cmd, check=True)

        if is_agentic:
            new_runs = snapshot_runs()
            created = [report_root / r for r in new_runs - existing_runs]
            for run_dir in created:
                artifacts = load_agentic_artifacts(run_dir)
                for key, items in artifacts.items():
                    agentic_artifacts[key].extend(items)
            existing_runs = new_runs
    
    # Handle ground truth
    if has_ground_truth:
        if not is_agentic:
            raise ValueError("Ground truth grading is only supported for agentic_coding benchmarks")
        
        print("Grading ground truth...")
        gt_scores = grade_ground_truth(bench_name, args.question_id, args.question_source, args.parallel)
        
        # Track artifacts for ground truth runs
        new_runs = snapshot_runs()
        created = [report_root / r for r in new_runs - existing_runs]
        for run_dir in created:
            artifacts = load_agentic_artifacts(run_dir)
            for key, items in artifacts.items():
                agentic_artifacts[key].extend(items)
        existing_runs = new_runs
        
        # Append ground truth scores to the temp file
        print(f"Writing {len(gt_scores)} ground truth scores to {temp_file}")
        with open(temp_file, 'a') as f:
            for question_id, score in gt_scores.items():
                judgment = {
                    'question_id': question_id,
                    'model': 'ground_truth',
                    'score': score
                }
                f.write(json.dumps(judgment) + '\n')

# Read all output files and collect scores by question_id and model
# Structure: {(question_id, model): [score1, score2, score3, score4, score5]}
scores_by_question_model = defaultdict(list)

for temp_file in temp_files:
    with open(temp_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            judgment = json.loads(line)
            question_id = judgment['question_id']
            model = judgment['model']
            score = judgment['score']
            scores_by_question_model[(question_id, model)].append(score)

print(f"\nCollected scores for {len(scores_by_question_model)} (question_id, model) pairs")
for (qid, model), scores in scores_by_question_model.items():
    print(f"  {model} - {qid}: {scores}")

def agentic_flaky_detail(key: tuple[str, str]) -> dict:
    entries = agentic_artifacts.get(key, [])
    failing_sets = []
    per_run = []
    for entry in entries:
        report_file = Path(entry.get("report_file", ""))
        fix_log_file = Path(entry.get("fix_log_file", ""))
        failing_tests = extract_failing_tests(report_file) if report_file.exists() else []
        failing_sets.append({name for name, _, _ in failing_tests})
        per_run.append({
            "eval_id": entry.get("eval_id", ""),
            "report_file": str(report_file),
            "fix_log_file": str(fix_log_file),
            "failing_tests": failing_tests
        })
    return {
        "has_flaky_tests": len({tuple(sorted(s)) for s in failing_sets if s}) > 1 if failing_sets else False,
        "per_run": per_run,
    }

# Identify questions with different scores
flaky_questions = []
all_flaky_tests = set()  # Track all flaky test names across all questions

for (question_id, model), scores in scores_by_question_model.items():
    if len(set(scores)) > 1:  # If not all scores are the same
        detail = agentic_flaky_detail((question_id, model)) if is_agentic else {}
        flaky_questions.append({
            'question_id': question_id,
            'model': model,
            'scores': scores,
            'detail': detail
        })
        
        # Collect flaky tests for this question
        if is_agentic and detail.get("per_run"):
            failing_sets = []
            for run_info in detail["per_run"]:
                failing_tests = run_info.get("failing_tests", [])
                failing_sets.append({name for name, _, _ in failing_tests})
            
            # A test is flaky if it appears in some failing sets but not all
            # (either fails sometimes or the set of failing tests changes)
            if len({tuple(sorted(s)) for s in failing_sets if s}) > 1:
                # Collect all test names that appear in at least one run
                all_tests_in_question = set()
                for s in failing_sets:
                    all_tests_in_question.update(s)
                # A test is flaky if it doesn't appear consistently
                for test_name in all_tests_in_question:
                    test_appears = [test_name in s for s in failing_sets]
                    if not all(test_appears) or (len(failing_sets) > 0 and not any(test_appears)):
                        all_flaky_tests.add(test_name)

# Output results
if flaky_questions:
    print("\n" + "="*80)
    print("FLAKY QUESTIONS DETECTED:")
    print("="*80)
    for item in flaky_questions:
        print(f"\nQuestion ID: {item['question_id']}")
        print(f"Model: {item['model']}")
        print(f"Scores across {num_iterations} runs: {item['scores']}")
        print(f"Unique scores: {set(item['scores'])}")
        if is_agentic:
            detail = item.get('detail', {})
            if detail.get("per_run"):
                print("Run artifacts:")
                for run_info in detail["per_run"]:
                    eval_id = run_info.get("eval_id", "")
                    report_file = run_info.get("report_file", "")
                    fix_log_file = run_info.get("fix_log_file", "")
                    failing_tests = run_info.get("failing_tests", [])
                    print(f"  eval_id: {eval_id}")
                    print(f"    report: {report_file}")
                    print(f"    fix log: {fix_log_file}")
                    if failing_tests:
                        print("    failing tests:")
                        for name, test_phase, fix_phase in failing_tests:
                            print(f"      {name}: {test_phase} -> {fix_phase}")
                    else:
                        print("    failing tests: none recorded")
            if detail.get("has_flaky_tests"):
                print("    Note: test-level flakiness detected across runs.")
else:
    print("\n" + "="*80)
    print(f"No flaky questions detected - all scores were consistent across {num_iterations} runs")
    print("="*80)

# Print all flaky tests at the end
if is_agentic and all_flaky_tests:
    print("\n" + "="*80)
    print("ALL FLAKY TESTS ACROSS ALL QUESTIONS:")
    print("="*80)
    for test_name in sorted(all_flaky_tests):
        print(f"  {test_name}")
    print(f"\nTotal flaky tests: {len(all_flaky_tests)}")
    print("="*80)
