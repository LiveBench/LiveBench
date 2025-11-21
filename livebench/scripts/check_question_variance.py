import subprocess
import argparse
import json
import os
from collections import defaultdict
import threading

parser = argparse.ArgumentParser(description='Check the variance of model answers and grading across multiple runs')
parser.add_argument('--model', required=True, nargs='+', help='Name of the model(s) to check')
parser.add_argument('--bench-name', required=True, help='Name of the benchmark to check')
parser.add_argument('--question-id', required=False, nargs='+', help='Question IDs to check')
parser.add_argument('--question-source', required=False, default='huggingface', help='Question source')
parser.add_argument('--parallel-requests', required=False, type=int, default=1, help='Number of parallel API request threads for gen_api_answer')
parser.add_argument('--parallel-grading', required=False, type=int, default=1, help='Number of parallel grading threads for gen_ground_truth_judgment')
parser.add_argument('--num-iterations', required=False, type=int, default=3, help='Number of iterations to run')

args = parser.parse_args()

num_iterations = args.num_iterations

# Create or use existing base output directory
base_dir = 'variance_check'
os.makedirs(base_dir, exist_ok=True)
print(f"Output directory: {base_dir}")

# Create directory structure and file paths for each model and iteration
iteration_data = []
all_answer_files = []
all_judgment_files = []

for model in args.model:
    model_dir = os.path.join(base_dir, model)
    os.makedirs(model_dir, exist_ok=True)
    
    for i in range(num_iterations):
        iter_dir = os.path.join(model_dir, f'iter{i}')
        os.makedirs(iter_dir, exist_ok=True)
        
        # Create file paths within the iteration directory
        # Name answer file after the model (required by load_model_answers)
        answer_file = os.path.join(iter_dir, f'{model}.jsonl')
        judgment_file = os.path.join(iter_dir, 'judgments.jsonl')
        answer_log = os.path.join(iter_dir, 'answer.log')
        judgment_log = os.path.join(iter_dir, 'judgment.log')
        
        all_answer_files.append(answer_file)
        all_judgment_files.append(judgment_file)
        
        iteration_data.append({
            'index': i,
            'model': model,
            'answer_file': answer_file,
            'judgment_file': judgment_file,
            'answer_log': answer_log,
            'judgment_log': judgment_log
        })


def run_iteration(data):
    """Run a single iteration for a single model: gen_api_answer followed by gen_ground_truth_judgment"""
    i = data['index']
    model = data['model']
    answer_file = data['answer_file']
    judgment_file = data['judgment_file']
    answer_log = data['answer_log']
    judgment_log = data['judgment_log']
    
    print(f"\n[Iteration {i+1}, Model {model}] Starting...")
    print(f"[Iteration {i+1}, Model {model}]   Answer file: {answer_file}")
    print(f"[Iteration {i+1}, Model {model}]   Judgment file: {judgment_file}")
    print(f"[Iteration {i+1}, Model {model}]   Answer log: {answer_log}")
    print(f"[Iteration {i+1}, Model {model}]   Judgment log: {judgment_log}")
    
    # Build gen_api_answer command (single model only)
    answer_cmd = [
        'python', 'gen_api_answer.py',
        '--bench-name', args.bench_name,
        '--model', model,
        '--answer-file', answer_file,
        '--parallel', str(args.parallel_requests),
        '--resume',
    ]
    
    if args.question_id:
        answer_cmd.extend(['--question-id', *args.question_id])
    
    if args.question_source:
        answer_cmd.extend(['--question-source', args.question_source])
    
    # Run gen_api_answer
    print(f"[Iteration {i+1}, Model {model}] Running answer generation...")
    with open(answer_log, 'w') as f_answer_log:
        answer_proc = subprocess.run(
            answer_cmd,
            stdout=f_answer_log,
            stderr=subprocess.STDOUT,
            text=True
        )
    
    if answer_proc.returncode != 0:
        print(f"[Iteration {i+1}, Model {model}] ERROR: Answer generation failed with return code {answer_proc.returncode}")
        print(f"[Iteration {i+1}, Model {model}]   Check log file: {answer_log}")
        return
    
    print(f"[Iteration {i+1}, Model {model}] Answer generation completed successfully")
    
    # Build gen_ground_truth_judgment command (single model only)
    judgment_cmd = [
        'python', 'gen_ground_truth_judgment.py',
        '--bench-name', args.bench_name,
        '--model', model,
        '--answer-file', answer_file,
        '--output-file', judgment_file,
        '--parallel', str(args.parallel_grading),
        '--resume',
    ]
    
    if args.question_id:
        judgment_cmd.extend(['--question-id', *args.question_id])
    
    if args.question_source:
        judgment_cmd.extend(['--question-source', args.question_source])
    
    # Run gen_ground_truth_judgment
    print(f"[Iteration {i+1}, Model {model}] Running grading...")
    with open(judgment_log, 'w') as f_judgment_log:
        judgment_proc = subprocess.run(
            judgment_cmd,
            stdout=f_judgment_log,
            stderr=subprocess.STDOUT,
            text=True
        )
    
    if judgment_proc.returncode != 0:
        print(f"[Iteration {i+1}, Model {model}] ERROR: Grading failed with return code {judgment_proc.returncode}")
        print(f"[Iteration {i+1}, Model {model}]   Check log file: {judgment_log}")
        return
    
    print(f"[Iteration {i+1}, Model {model}] Grading completed successfully")


print(f"\n{'='*80}")
print(f"Starting {num_iterations} iterations × {len(args.model)} models = {len(iteration_data)} parallel runs...")
print(f"{'='*80}\n")

# Run all iterations in parallel using threads
threads = []
for data in iteration_data:
    thread = threading.Thread(target=run_iteration, args=(data,))
    thread.start()
    threads.append(thread)

# Wait for all threads to complete
for thread in threads:
    thread.join()

print(f"\n{'='*80}")
print("All iterations completed. Analyzing results...")
print(f"{'='*80}\n")

# Read all judgment files and collect scores by question_id and model
# Structure: {(question_id, model): [score1, score2, ...]}
scores_by_question_model = defaultdict(list)
answers_by_question_model = defaultdict(list)

# Also collect the actual answers for comparison
for i, (answer_file, judgment_file) in enumerate(zip(all_answer_files, all_judgment_files)):
    # Read answers
    if os.path.exists(answer_file):
        with open(answer_file, 'r') as f:
            for line in f:
                try:
                    answer = json.loads(line)
                    question_id = answer['question_id']
                    model = answer['model_id']
                    # Store the answer text (first turn of first choice)
                    answer_text = answer['choices'][0]['turns'][0] if answer['choices'] else ''
                    answers_by_question_model[(question_id, model)].append((i, answer_text))
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Error reading answer from iteration {i}: {e}")
    
    # Read judgments
    if os.path.exists(judgment_file):
        with open(judgment_file, 'r') as f:
            for line in f:
                try:
                    judgment = json.loads(line)
                    question_id = judgment['question_id']
                    model = judgment['model']
                    score = judgment['score']
                    scores_by_question_model[(question_id, model)].append(score)
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Error reading judgment from iteration {i}: {e}")

# Identify questions with score variance
variant_questions = []
for (question_id, model), scores in scores_by_question_model.items():
    if len(set(scores)) > 1:  # If not all scores are the same
        variant_questions.append({
            'question_id': question_id,
            'model': model,
            'scores': scores
        })

# Identify questions with different answers
variant_answers = []
for (question_id, model), answer_data in answers_by_question_model.items():
    # Extract just the answer texts
    answers = [text for _, text in answer_data]
    # Check if all answers are identical
    if len(set(answers)) > 1:
        variant_answers.append({
            'question_id': question_id,
            'model': model,
            'num_unique_answers': len(set(answers)),
            'iterations': [iter_num for iter_num, _ in answer_data],
            'answers': answers
        })

# Output results
print("\n" + "="*80)
print("VARIANCE ANALYSIS RESULTS")
print(f"Models: {', '.join(args.model)}")
print(f"Iterations: {num_iterations}")
print("="*80)

# Print all scores for all questions
print("\n" + "-"*80)
print(f"ALL SCORES ({len(scores_by_question_model)} question-model pairs):")
print("-"*80)
for (question_id, model), scores in sorted(scores_by_question_model.items()):
    print(f"\nQuestion ID: {question_id}")
    print(f"Model: {model}")
    print(f"Scores across {num_iterations} runs: {scores}")
    if len(set(scores)) > 1:
        print(f"  ⚠️  VARIANCE DETECTED - Unique scores: {set(scores)}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Models tested: {', '.join(args.model)}")
print(f"Iterations per model: {num_iterations}")
print(f"Total runs: {len(iteration_data)}")
print(f"Total question-model pairs evaluated: {len(scores_by_question_model)}")
print(f"Question-model pairs with answer variance: {len(variant_answers)}")
print(f"Question-model pairs with score variance: {len(variant_questions)}")
print(f"Output directory: {base_dir}")
print("="*80)

print(f"\nAll files preserved in: {base_dir}")

