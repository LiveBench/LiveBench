#!/usr/bin/env python3
import json
import os
import glob
import re
import concurrent.futures
import argparse
import subprocess
from collections import defaultdict, Counter
import traceback
from livebench.model.api_model_config import get_model_config

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Find syntax errors in model outputs')
parser.add_argument('--model', nargs='+', help='Only evaluate files for these model names')
parser.add_argument('--only-overlapping', action='store_true', help='Only include question IDs that have all the error strings in ALL_ERROR_STRINGS')
parser.add_argument('--only-incorrect', action='store_true', help='Only print the error question IDs where the score was 0')
parser.add_argument('--rerun', action='store_true', help='Rerun questions with errors for each model')
args = parser.parse_args()

# Paths to directories (relative to LiveBench/livebench/)
jsonl_dirs = ["data/live_bench/agentic_coding/python/model_answer","data/live_bench/agentic_coding/javascript/model_answer","data/live_bench/agentic_coding/typescript/model_answer"]
trajectories_dir = "agentic_code_runner/data/trajectories"
questions_files = ["data/live_bench/agentic_coding/python/question.jsonl","data/live_bench/agentic_coding/javascript/question.jsonl","data/live_bench/agentic_coding/typescript/question.jsonl"]
judgment_files = ["data/live_bench/agentic_coding/python/model_judgment/ground_truth_judgment.jsonl","data/live_bench/agentic_coding/javascript/model_judgment/ground_truth_judgment.jsonl","data/live_bench/agentic_coding/typescript/model_judgment/ground_truth_judgment.jsonl"]

# Get the script directory and construct absolute paths
script_dir = os.path.dirname(os.path.abspath(__file__))
livebench_dir = os.path.dirname(script_dir)  # Go up one level from scripts/ to livebench/

# Convert relative paths to absolute paths
jsonl_dirs = [os.path.join(livebench_dir, path) for path in jsonl_dirs]
trajectories_dir = os.path.join(livebench_dir, trajectories_dir)
questions_files = [os.path.join(livebench_dir, path) for path in questions_files]
judgment_files = [os.path.join(livebench_dir, path) for path in judgment_files]

# Load valid question IDs
print("Loading valid question IDs...")
valid_question_ids = set()
for questions_file in questions_files:
    try:
        with open(questions_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    question_id = data.get('question_id')
                    if question_id:
                        valid_question_ids.add(str(question_id))
                except json.JSONDecodeError:
                    continue
        print(f"Loaded {len(valid_question_ids)} valid question IDs from " + questions_file)
    except FileNotFoundError:
        print(f"Warning: Questions file not found at {questions_file}")
        print("Proceeding without question ID validation")
        valid_question_ids = None
        break

# Load model judgments
print("Loading model judgments...")
model_judgments = {}  # (model_name, question_id) -> score
total_judgments_loaded = 0

for judgment_file in judgment_files:
    try:
        language = judgment_file.split('/')[-3]  # Extract language from path
        judgments_for_language = 0
        with open(judgment_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    question_id = data.get('question_id')
                    model = data.get('model')
                    score = data.get('score')
                    if question_id and model is not None and score is not None:
                        model_judgments[(model, str(question_id))] = score
                        judgments_for_language += 1
                        total_judgments_loaded += 1
                except json.JSONDecodeError:
                    continue
        print(f"Loaded {judgments_for_language} model judgments from {language}")
    except FileNotFoundError:
        print(f"Warning: Model judgment file not found at {judgment_file}")

print(f"Total model judgments loaded: {total_judgments_loaded}")

# Set maximum number of workers for parallel processing
MAX_WORKERS = 20

SWEAGENT_ERROR_STATUSES = [
    'LimitsExceeded',
]

OTHER_ERROR_STRINGS = [
    # '\\"instance_cost\\": 3'
    "This endpoint's maximum context length"
]

ALL_ERROR_STRINGS = SWEAGENT_ERROR_STATUSES + OTHER_ERROR_STRINGS


# Dictionary to store model -> (run_id, question_id) pairs
model_pairs = defaultdict(list)

# Results dictionary to store model -> (run_id, question_id) pairs with errors
model_errors = defaultdict(list)

# Dictionary to track how many times each (model, run_id, question_id, error) tuple appears
found_errors = Counter()

# Dictionary to track which error strings each question_id has encountered per model
model_question_errors = defaultdict(lambda: defaultdict(set))

# Function to process a single JSONL file
def process_jsonl_file(jsonl_file, valid_question_ids=None):
    local_model_pairs = []
    local_model_errors = []
    local_found_errors = Counter()
    # Track which error strings each question_id has encountered
    local_question_errors = defaultdict(set)
    
    model_name = os.path.basename(jsonl_file).replace(".jsonl", "")
    print(f"Processing {model_name}...")

    # Read the JSONL file
    with open(jsonl_file, 'r') as f:
        for line_num, line in enumerate(f):
            try:
                data = json.loads(line)
                run_id = str(data.get('run_id'))
                question_id = str(data.get('question_id'))
                if run_id and question_id:
                    local_model_pairs.append((run_id, question_id))
                    
                    # Skip processing if question_id is not in valid questions list
                    if valid_question_ids is not None and question_id not in valid_question_ids:
                        continue
                    
                    # Check for error pattern in the original JSONL line
                    for error_string in ALL_ERROR_STRINGS:
                        if error_string in line:
                            error_found = error_string
                            # Track which error strings this question has
                            local_question_errors[question_id].add(error_found)
                            # Create a unique identifier for this error
                            error_id = (model_name, run_id, question_id, error_found)
                            # Increment the counter for this error
                            local_found_errors[error_id] += 1
                            # Only add to model_errors if this is the first occurrence
                            if local_found_errors[error_id] == 1:
                                local_model_errors.append((run_id, question_id, f"JSONL:{error_found}"))
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {jsonl_file}, line: {line_num}")
                continue
    return model_name, local_model_pairs, local_model_errors, local_found_errors, local_question_errors

# Process JSONL files in parallel
if args.model:
    # Only process specified models
    jsonl_files = []
    for jsonl_dir in jsonl_dirs:
        jsonl_files += [os.path.join(jsonl_dir, f"{get_model_config(model_name).display_name}.jsonl") for model_name in args.model]
    # Filter out non-existent files
    jsonl_files = [f for f in jsonl_files if os.path.exists(f)]
    print(f"Found {len(jsonl_files)} JSONL files to process for specified models: {', '.join(args.model)}")
else:
    # Process all JSONL files
    jsonl_files = []
    for jsonl_dir in jsonl_dirs:
        jsonl_files += glob.glob(os.path.join(jsonl_dir, "*.jsonl"))
    print(f"Found {len(jsonl_files)} JSONL files to process")

with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # Submit all JSONL files for processing
    future_to_file = {executor.submit(process_jsonl_file, jsonl_file, valid_question_ids): jsonl_file for jsonl_file in jsonl_files}
    
    # Process results as they complete
    for future in concurrent.futures.as_completed(future_to_file):
        jsonl_file = future_to_file[future]
        try:
            model_name, local_pairs, local_errors, local_found, local_question_errors = future.result()
            model_pairs[model_name].extend(local_pairs)
            model_errors[model_name].extend(local_errors)
            # Update the counter with values from local_found
            for error_id, count in local_found.items():
                found_errors[error_id] += count
            # Update the question errors tracking
            for question_id, error_set in local_question_errors.items():
                model_question_errors[model_name][question_id].update(error_set)
        except Exception as exc:
            print(f"Processing {jsonl_file} generated an exception: {exc}")
            traceback.print_exc()       

# Filter results if --only-overlapping is specified
if args.only_overlapping:
    print("\n" + "="*50)
    print("FILTERING: Only including question IDs with ALL error strings")
    print("="*50)
    
    # Get all non-empty error strings from ALL_ERROR_STRINGS
    non_empty_error_strings = set(error for error in ALL_ERROR_STRINGS if error.strip())
    print(f"Total error strings to check: {len(non_empty_error_strings)}")
    
    if non_empty_error_strings:
        # Filter model_errors to only include question IDs that have all error strings
        filtered_model_errors = defaultdict(list)
        
        for model_name, error_pairs in model_errors.items():
            if not error_pairs:
                continue
                
            # Group errors by (run_id, question_id)
            question_errors_map = defaultdict(list)
            for run_id, question_id, error in error_pairs:
                question_errors_map[(run_id, question_id)].append(error)
            
            # Check each question to see if it has all error strings
            for (run_id, question_id), errors in question_errors_map.items():
                # Extract the actual error strings (remove JSONL: prefix if present)
                actual_errors = set()
                for error in errors:
                    if error.startswith("JSONL:"):
                        actual_errors.add(error[6:])  # Remove "JSONL:" prefix
                    else:
                        actual_errors.add(error)
                
                # Check if this question has all the error strings
                if non_empty_error_strings.issubset(actual_errors):
                    # Add all errors for this question to filtered results
                    for error in errors:
                        filtered_model_errors[model_name].append((run_id, question_id, error))
        
        # Replace model_errors with filtered results
        model_errors = filtered_model_errors
        
        # Print filtering results
        total_filtered = sum(len(errors) for errors in filtered_model_errors.values())
        print(f"After filtering: {total_filtered} error entries remain")
    else:
        print("No non-empty error strings found, no filtering applied")

# Print results
print("\n" + "="*50)
if args.only_incorrect:
    print("RESULTS: Question IDs with errors WHERE SCORE = 0")
else:
    print("RESULTS: Question IDs with errors")
print("="*50)

for model, error_pairs in model_errors.items():
    if not error_pairs:
        continue
    
    # Filter error_pairs for only incorrect answers if --only-incorrect is specified
    if args.only_incorrect:
        filtered_error_pairs = []
        for run_id, question_id, error in error_pairs:
            judgment_key = (model, question_id)
            if judgment_key in model_judgments:
                score = model_judgments[judgment_key]
                if score == 0:  # Only include questions with score = 0
                    filtered_error_pairs.append((run_id, question_id, error))
        error_pairs = filtered_error_pairs
        
        # Skip this model if no incorrect answers with errors
        if not error_pairs:
            continue
        
    print(f"\nModel: {model}")
    
    # Count unique question IDs with errors
    unique_question_ids = set()
    for run_id, question_id, error in error_pairs:
        unique_question_ids.add(question_id)
    
    # Calculate accuracy statistics
    total_questions = len(model_pairs[model])
    questions_with_errors = len(unique_question_ids)
    
    # Get judgment scores for this model
    correct_answers = 0
    total_judged = 0
    correct_with_errors = 0
    incorrect_with_errors = 0
    
    for run_id, question_id in model_pairs[model]:
        judgment_key = (model, question_id)
        if judgment_key in model_judgments:
            total_judged += 1
            score = model_judgments[judgment_key]
            if score > 0:  # Assuming score > 0 means correct
                correct_answers += 1
                if question_id in unique_question_ids:
                    correct_with_errors += 1
            else:
                if question_id in unique_question_ids:
                    incorrect_with_errors += 1
    
    accuracy = (correct_answers / total_judged * 100) if total_judged > 0 else 0
    
    print(f"Total questions: {total_questions}")
    print(f"Questions with judgments: {total_judged}")
    print(f"Correct answers: {correct_answers}/{total_judged} ({accuracy:.1f}%)")
    print(f"Questions with errors: {questions_with_errors}/{total_questions}")
    print(f"Correct answers with errors: {correct_with_errors}")
    print(f"Incorrect answers with errors: {incorrect_with_errors}")
    
    if questions_with_errors > 0:
        error_accuracy = (correct_with_errors / questions_with_errors * 100) if questions_with_errors > 0 else 0
        print(f"Accuracy among questions with errors: {correct_with_errors}/{questions_with_errors} ({error_accuracy:.1f}%)")

    if args.only_overlapping:
        # For --only-overlapping, show simplified output since all question IDs have all error strings
        run_id_to_question_ids = defaultdict(set)
        question_id_counts = defaultdict(int)
        
        for run_id, question_id, error in error_pairs:
            run_id_to_question_ids[run_id].add(question_id)
            # Count the maximum occurrences for this question ID across all error types
            error_type = error[6:] if error.startswith("JSONL:") else error
            error_id = (model, run_id, question_id, error_type)
            count = found_errors[error_id]
            question_id_counts[question_id] = max(question_id_counts[question_id], count)
        
        for run_id, question_ids in run_id_to_question_ids.items():
            print(f"  Run ID: {run_id}")
            # Show question IDs with their maximum counts and judgment scores
            question_ids_with_info = []
            for qid in sorted(question_ids):
                count = question_id_counts[qid]
                judgment_key = (model, qid)
                score = model_judgments.get(judgment_key, "N/A")
                score_str = f"score:{score}" if score != "N/A" else "score:N/A"
                # question_ids_with_info.append(f"{qid}({count},{score_str})")
                question_ids_with_info.append(f"{qid}")
            
            if args.only_incorrect:
                print(f"    Question IDs (all have ALL error strings AND score=0): {' '.join(question_ids_with_info)}")
            else:
                print(f"    Question IDs (all have ALL error strings): {' '.join(question_ids_with_info)}")
    # elif args.only_incorrect:
    #     # For --only-incorrect, show simplified output with just question IDs and scores
    #     run_id_to_question_ids = defaultdict(set)
    #     question_id_counts = defaultdict(int)
        
    #     for run_id, question_id, error in error_pairs:
    #         run_id_to_question_ids[run_id].add(question_id)
    #         # Count the maximum occurrences for this question ID across all error types
    #         error_type = error[6:] if error.startswith("JSONL:") else error
    #         error_id = (model, run_id, question_id, error_type)
    #         count = found_errors[error_id]
    #         question_id_counts[question_id] = max(question_id_counts[question_id], count)
        
    #     for run_id, question_ids in run_id_to_question_ids.items():
    #         print(f"  Run ID: {run_id}")
    #         # Show question IDs with their maximum counts and judgment scores (all should be score=0)
    #         question_ids_with_info = []
    #         for qid in sorted(question_ids):
    #             count = question_id_counts[qid]
    #             judgment_key = (model, qid)
    #             score = model_judgments.get(judgment_key, "N/A")
    #             score_str = f"score:{score}" if score != "N/A" else "score:N/A"
    #             # question_ids_with_info.append(f"{qid}({count},{score_str})")
    #             question_ids_with_info.append(f"{qid}")
            
    #         print(f"    Question IDs (score=0): {' '.join(question_ids_with_info)}")
    else:
        # Original detailed output for normal mode
        run_id_to_errors = defaultdict(lambda: defaultdict(list))
        error_counts = defaultdict(int)
        
        for run_id, question_id, error in error_pairs:
            # Prefix to indicate where the error was found
            if error.startswith("JSONL:"):
                source = "JSONL file"
                error_type = error[6:]  # Remove the "JSONL:" prefix
            else:
                source = "Trajectory file"
                error_type = error
                
            # Get the count for this error
            error_id = (model, run_id, question_id, error_type)
            count = found_errors[error_id]
            error_counts[(source, error_type, question_id)] = count
            run_id_to_errors[run_id][(source, error_type)].append(question_id)
        
        for run_id, error_to_question_ids in run_id_to_errors.items():
            print(f"  Run ID: {run_id}")
            overlap = None
            for (source, error), question_ids in error_to_question_ids.items():
                # Display question IDs with their error counts and judgment scores
                question_ids_with_info = []
                for qid in question_ids:
                    count = error_counts[(source, error, qid)]
                    judgment_key = (model, qid)
                    score = model_judgments.get(judgment_key, "N/A")
                    score_str = f"score:{score}" if score != "N/A" else "score:N/A"
                    # question_ids_with_info.append(f"{qid}({count},{score_str})")
                    question_ids_with_info.append(f"{qid}")
                
                print(f"    {error}: {' '.join(question_ids_with_info)}")
                
                if overlap is None:
                    overlap = set(question_ids)
                else:
                    overlap = overlap.intersection(set(question_ids))
            
            if len(error_to_question_ids) > 1 and overlap is not None and len(overlap) > 0:
                # Display overlapping question IDs with their counts and judgment scores
                overlap_with_info = []
                for qid in overlap:
                    max_count = max(error_counts[(source, error, qid)] for source, error in error_to_question_ids.keys())
                    judgment_key = (model, qid)
                    score = model_judgments.get(judgment_key, "N/A")
                    score_str = f"score:{score}" if score != "N/A" else "score:N/A"
                    # overlap_with_info.append(f"{qid}({max_count},{score_str})")
                    overlap_with_info.append(f"{qid}")
                
                print(f"  Overlapping question IDs: {' '.join(overlap_with_info)}")

# Print summary statistics
print("\n" + "="*50)
if args.only_incorrect:
    print("SUMMARY STATISTICS (ONLY INCORRECT ANSWERS WITH ERRORS)")
else:
    print("SUMMARY STATISTICS")
print("="*50)

summary_stats = []
for model, error_pairs in model_errors.items():
    if not error_pairs:
        continue
    
    # Filter error_pairs for only incorrect answers if --only-incorrect is specified
    if args.only_incorrect:
        filtered_error_pairs = []
        for run_id, question_id, error in error_pairs:
            judgment_key = (model, question_id)
            if judgment_key in model_judgments:
                score = model_judgments[judgment_key]
                if score == 0:  # Only include questions with score = 0
                    filtered_error_pairs.append((run_id, question_id, error))
        error_pairs = filtered_error_pairs
        
        # Skip this model if no incorrect answers with errors
        if not error_pairs:
            continue
    
    # Count unique question IDs with errors
    unique_question_ids = set()
    for run_id, question_id, error in error_pairs:
        unique_question_ids.add(question_id)
    
    # Calculate accuracy statistics
    total_questions = len(model_pairs[model])
    questions_with_errors = len(unique_question_ids)
    
    # Get judgment scores for this model
    correct_answers = 0
    total_judged = 0
    correct_with_errors = 0
    
    for run_id, question_id in model_pairs[model]:
        judgment_key = (model, question_id)
        if judgment_key in model_judgments:
            total_judged += 1
            score = model_judgments[judgment_key]
            if score > 0:  # Assuming score > 0 means correct
                correct_answers += 1
                if question_id in unique_question_ids:
                    correct_with_errors += 1
    
    accuracy = (correct_answers / total_judged * 100) if total_judged > 0 else 0
    error_rate = (questions_with_errors / total_questions * 100) if total_questions > 0 else 0
    error_accuracy = (correct_with_errors / questions_with_errors * 100) if questions_with_errors > 0 else 0
    
    summary_stats.append({
        'model': model,
        'total_questions': total_questions,
        'total_judged': total_judged,
        'accuracy': accuracy,
        'questions_with_errors': questions_with_errors,
        'error_rate': error_rate,
        'correct_with_errors': correct_with_errors,
        'error_accuracy': error_accuracy
    })

# Sort by error rate (descending)
summary_stats.sort(key=lambda x: x['error_rate'], reverse=True)

print(f"{'Model':<30} {'Total':<6} {'Judged':<6} {'Acc%':<6} {'Errors':<7} {'Err%':<6} {'CorrectErr':<10} {'ErrAcc%':<7}")
print("-" * 80)
for stats in summary_stats:
    print(f"{stats['model']:<30} {stats['total_questions']:<6} {stats['total_judged']:<6} {stats['accuracy']:<6.1f} {stats['questions_with_errors']:<7} {stats['error_rate']:<6.1f} {stats['correct_with_errors']:<10} {stats['error_accuracy']:<7.1f}")

print("\nDone!") 

# Rerun questions with errors if --rerun flag is set
if args.rerun:
    print("\n" + "="*50)
    if args.only_incorrect:
        print("RERUNNING QUESTIONS WITH ERRORS (ONLY INCORRECT ANSWERS)")
    else:
        print("RERUNNING QUESTIONS WITH ERRORS")
    print("="*50)
    
    # Collect all question IDs with errors for each model
    model_to_question_ids = defaultdict(set)
    for model, error_pairs in model_errors.items():
        if not error_pairs:
            continue
        
        # Filter error_pairs for only incorrect answers if --only-incorrect is specified
        if args.only_incorrect:
            filtered_error_pairs = []
            for run_id, question_id, error in error_pairs:
                judgment_key = (model, question_id)
                if judgment_key in model_judgments:
                    score = model_judgments[judgment_key]
                    if score == 0:  # Only include questions with score = 0
                        filtered_error_pairs.append((run_id, question_id, error))
            error_pairs = filtered_error_pairs
            
        for run_id, question_id, error in error_pairs:
            # Only include question IDs that are in the valid questions list
            if valid_question_ids is None or question_id in valid_question_ids:
                model_to_question_ids[model].add(question_id)
    
    # Rerun each model with its error questions
    for model, question_ids in model_to_question_ids.items():
        if not question_ids:
            continue
            
        question_ids_str = " ".join(sorted(list(question_ids)))
        print(f"\nRerunning {len(question_ids)} questions for model {model}")
        print(f"Question IDs: {question_ids_str}")
        
        # Construct and run the command
        cmd = [
            "python", "run_livebench.py",
            "--model", model,
            "--bench-name", "live_bench/agentic_coding",
            "--question-source", "jsonl",
            "--mode", "parallel",
            "--parallel-requests", "10",
            "--parallel-grading", "10",
            "--question-id"] + list(question_ids)
        
        print(f"Running command: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, text=True)
            print(f"Command completed successfully for model {model}")
        except subprocess.CalledProcessError as e:
            print(f"Error running command for model {model}: {e}")
            print(f"Error output: {e.stderr}")
