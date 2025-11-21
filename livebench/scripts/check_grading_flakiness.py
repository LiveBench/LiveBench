import subprocess
import tempfile
import argparse
import json
import os
from collections import defaultdict

parser = argparse.ArgumentParser(description='Check the grading flakiness of a task for a given model\'s answers')
parser.add_argument('--model', required=True, nargs='+', help='Name of the model(s) to check')
parser.add_argument('--bench-name', required=True, help='Name of the benchmark to check')
parser.add_argument('--question-id', required=False, nargs='+', help='Question IDs to check')
parser.add_argument('--question-source', required=False, help='Question source')
parser.add_argument('--parallel', required=False, type=int, help='Number of parallel grading threads')

args = parser.parse_args()

num_iterations = 5

# Create 5 temporary files
temp_files = []
for i in range(num_iterations):
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
    temp_files.append(temp_file.name)
    temp_file.close()

# Run gen_ground_truth_judgment.py 5 times
for i, temp_file in enumerate(temp_files):
    print(f"Running grading iteration {i+1}/{num_iterations}...")
    cmd = [
        'python', 'gen_ground_truth_judgment.py',
        '--bench-name', args.bench_name,
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

# Identify questions with different scores
flaky_questions = []
for (question_id, model), scores in scores_by_question_model.items():
    if len(set(scores)) > 1:  # If not all scores are the same
        flaky_questions.append({
            'question_id': question_id,
            'model': model,
            'scores': scores
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
else:
    print("\n" + "="*80)
    print(f"No flaky questions detected - all scores were consistent across {num_iterations} runs")
    print("="*80)

# Clean up temporary files
for temp_file in temp_files:
    os.unlink(temp_file)
