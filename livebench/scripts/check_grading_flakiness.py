import argparse
import json
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from livebench.common import LIVE_BENCH_ROOT_PATH
from livebench.agentic_code_runner.eval.harness.report import Report
from livebench.agentic_code_runner.eval.harness.test_result import TestStatus

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

agentic_artifacts: dict[tuple[str, str], list[dict]] = defaultdict(list)

# Run gen_ground_truth_judgment.py 5 times
existing_runs = snapshot_runs() if is_agentic else set()
for i, temp_file in enumerate(temp_files):
    print(f"Running grading iteration {i+1}/{num_iterations}...")
    cmd = [
        'python', 'gen_ground_truth_judgment.py',
        '--bench-name', bench_name,
        '--model', *args.model,
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

# Read all output files and collect scores by question_id and model
# Structure: {(question_id, model): [score1, score2, score3, score4, score5]}
scores_by_question_model = defaultdict(list)

for temp_file in temp_files:
    with open(temp_file, 'r') as f:
        for line in f:
            judgment = json.loads(line)
            question_id = judgment['question_id']
            model = judgment['model']
            score = judgment['score']
            scores_by_question_model[(question_id, model)].append(score)

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
for (question_id, model), scores in scores_by_question_model.items():
    if len(set(scores)) > 1:  # If not all scores are the same
        detail = agentic_flaky_detail((question_id, model)) if is_agentic else {}
        flaky_questions.append({
            'question_id': question_id,
            'model': model,
            'scores': scores,
            'detail': detail
        })

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
