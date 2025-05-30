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
parser.add_argument('--only-overlapping', action='store_true', help='Display overlapping errors')
parser.add_argument('--rerun', action='store_true', help='Rerun questions with errors for each model')
args = parser.parse_args()

# Paths to directories
jsonl_dirs = ["data/live_bench/agentic_coding/python/model_answer","data/live_bench/agentic_coding/javascript/model_answer","data/live_bench/agentic_coding/typescript/model_answer"]
trajectories_dir = "agentic_code_runner/data/trajectories"
questions_files = ["data/live_bench/agentic_coding/python/question.jsonl","data/live_bench/agentic_coding/javascript/question.jsonl","data/live_bench/agentic_coding/typescript/question.jsonl"]

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

# Set maximum number of workers for parallel processing
MAX_WORKERS = 20

SWEAGENT_ERROR_STATUSES = [
    # 'exit_total_execution_time',
    # 'exit_command_timeout',
    # 'exit_context',
    # 'exit_api',
    # 'exit_environment_error',
    # 'exit_error',
    'exit_format',
    # 'exit_cost',
    # 'Bad gateway'
]

OTHER_ERROR_STRINGS = [
    # 'Please make sure your output precisely matches the following format:\n2025',
    # 'Unexpected argument(s): pattern'
    # "timeout after 300.0 seconds while running command 'submit'",
    # "list index out of range",
    # "Operation 'insert",
    # "Operation 'open",
    # "Operation 'create",
    # "Operation 'edit",
    # "Operation 'search",
    # "Operation 'find"
    # "Text '1:' not found in displayed lines"
    # "Operation 'python",
    # "Operation",
    # "Together_aiException"
    # "python manage.py runserver",
    # "action\\\": \\\"find_file",
    # "Failed to interrupt session",
    # "timeout after 300.0 seconds while running command",
    # "The command 'edit",
    # "This model's maximum context length",
    # "This request would exceed the rate limit for your organization",
    # "You exceeded your current quota, please check your plan and billing details",
    # "doesn't support tool_choice=required"
    # "git restore .gitignore",
    # "exceeds maximum input length",
    # "tool_choice=required",
    # "all elements in history must have a message",
    # "You exceeded your current quota",
    # "MistralException - Service unavailable.",
    # "'NoneType' object has no attribute 'keys'",
    # "invalid tool call provided",
    # "The length of your prompt exceeds the model's max input limit"
]

blocklist: list[str] = [
    "vim",
    "vi",
    "emacs",
    "nano",
    "nohup",
    "gdb",
    "less",
    "tail -f",
    "python -m venv",
    "python3 -m venv",
    "pip install",
    "npm install",
    "pnpm install",
    "playright install",
    "bash\n",
    "sh\n",
    "/bin/bash\n",
    "/bin/sh\n",
]
blocklist_standalone: list[str] = [
    "ipython",
    "nohup",
    "vi",
    "vim",
    "emacs",
    "nano",
    "su",
    "bash",
    "sh",
    "/bin/bash",
    "/bin/sh",
    "npm run dev",
    "npm run preview",
    "pnpm run dev",
    "python",
    "python3",
    "deactivate"
]

# OTHER_ERROR_STRINGS += [f"action\\\": \\\"{cmd} " for cmd in blocklist]
# OTHER_ERROR_STRINGS += [f"action\\\": \\\"{cmd}\n" for cmd in blocklist]
# OTHER_ERROR_STRINGS += [f" && {cmd} " for cmd in blocklist]
# OTHER_ERROR_STRINGS += [f" {cmd} && " for cmd in blocklist]
# OTHER_ERROR_STRINGS += [f" && {cmd}\n" for cmd in blocklist]
# OTHER_ERROR_STRINGS += [f" && {cmd}\\\"" for cmd in blocklist]
# OTHER_ERROR_STRINGS += [f"action\\\": \\\"{cmd}\\\"" for cmd in blocklist_standalone]
# OTHER_ERROR_STRINGS += [f"action\\\": \\\"{cmd}\n\\\"" for cmd in blocklist_standalone]

ALL_ERROR_STRINGS = SWEAGENT_ERROR_STATUSES + OTHER_ERROR_STRINGS

CONTENT_ERROR_REGEXES = {
    # 'multiple commands': r'(?:.*?<command>.*?<\/command>){2,}.*?'
    # 'function_call': '<function_call>'
}

# Dictionary to store model -> (run_id, question_id) pairs
model_pairs = defaultdict(list)

# Results dictionary to store model -> (run_id, question_id) pairs with errors
model_errors = defaultdict(list)

# Dictionary to track how many times each (model, run_id, question_id, error) tuple appears
found_errors = Counter()

# Function to process a single JSONL file
def process_jsonl_file(jsonl_file, valid_question_ids=None):
    local_model_pairs = []
    local_model_errors = []
    local_found_errors = Counter()
    
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
                            # Create a unique identifier for this error
                            error_id = (model_name, run_id, question_id, error_found)
                            # Increment the counter for this error
                            local_found_errors[error_id] += 1
                            # Only add to model_errors if this is the first occurrence
                            if local_found_errors[error_id] == 1:
                                local_model_errors.append((run_id, question_id, f"JSONL:{error_found}"))
                    if len(CONTENT_ERROR_REGEXES) > 0:
                        if 'history' in data:
                            content_pattern = r'"role"\s*:\s*"assistant"\s*,\s*"content"\s*:\s*"((?:\\.|[^"\\])*)(?:",|\"\s*})'
                            matches = re.finditer(content_pattern, data['history'])
                        elif 'trajectory' in data:
                            response_pattern = r'"response"\s*:\s*"((?:\\.|[^"\\])*)(?:",|\"\s*})'
                            matches = re.finditer(response_pattern, data['trajectory'])
                        else:
                            print('No history or trajectory found for model ' + model_name + ' question ' + question_id)
                            continue
                        remaining_poss_errors = set(CONTENT_ERROR_REGEXES.keys())
                        for i, match in enumerate(matches):
                            content_value = match.group(1)

                            for error_name, regex in CONTENT_ERROR_REGEXES.items():
                                if re.search(regex, content_value, re.DOTALL):
                                    error_id = (model_name, run_id, question_id, error_name)
                                    local_found_errors[error_id] += 1
                                    if local_found_errors[error_id] == 1:
                                        local_model_errors.append((run_id, question_id, error_name))
                                        remaining_poss_errors.remove(error_name)
                            
                            if len(remaining_poss_errors) == 0:
                                break
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {jsonl_file}, line: {line_num}")
                continue
    return model_name, local_model_pairs, local_model_errors, local_found_errors

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
            model_name, local_pairs, local_errors, local_found = future.result()
            model_pairs[model_name].extend(local_pairs)
            model_errors[model_name].extend(local_errors)
            # Update the counter with values from local_found
            for error_id, count in local_found.items():
                found_errors[error_id] += count
        except Exception as exc:
            print(f"Processing {jsonl_file} generated an exception: {exc}")
            traceback.print_exc()       

# Print results
print("\n" + "="*50)
print("RESULTS: Question IDs with errors")
print("="*50)

for model, error_pairs in model_errors.items():
    if not error_pairs:
        continue
        
    print(f"\nModel: {model}")
    print(f"Total pairs with errors: {len(error_pairs)}/{len(model_pairs[model])}")

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
            # Display question IDs with their error counts
            question_ids_with_counts = []
            for qid in question_ids:
                count = error_counts[(source, error, qid)]
                question_ids_with_counts.append(f"{qid}({count})")
            
            print(f"    {error}: {' '.join(question_ids_with_counts)}")
            
            if overlap is None:
                overlap = set(question_ids)
            else:
                overlap = overlap.intersection(set(question_ids))
        
        if len(error_to_question_ids) > 1 and overlap is not None and len(overlap) > 0:
            # Display overlapping question IDs with their counts for the most frequent error
            overlap_with_counts = []
            for qid in overlap:
                max_count = max(error_counts[(source, error, qid)] for source, error in error_to_question_ids.keys())
                overlap_with_counts.append(f"{qid}({max_count})")
            
            print(f"  Overlapping question IDs: {' '.join(overlap_with_counts)}")

print("\nDone!") 

# Rerun questions with errors if --rerun flag is set
if args.rerun:
    print("\n" + "="*50)
    print("RERUNNING QUESTIONS WITH ERRORS")
    print("="*50)
    
    # Collect all question IDs with errors for each model
    model_to_question_ids = defaultdict(set)
    for model, error_pairs in model_errors.items():
        if not error_pairs:
            continue
            
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
