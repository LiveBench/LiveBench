#!/usr/bin/env python3
import json
import os
import glob
import re
import concurrent.futures
import argparse
import subprocess
from collections import defaultdict
from tqdm import tqdm
import traceback

from livebench.model.api_model_config import get_model_config

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Find syntax errors in model outputs')
parser.add_argument('--model', nargs='+', help='Only evaluate files for these model names')
parser.add_argument('--only-overlapping', action='store_true', help='Display overlapping errors')
parser.add_argument('--rerun', action='store_true', help='Rerun questions with errors for each model')
args = parser.parse_args()

# Paths to directories
jsonl_dir = "data/live_bench/coding/agentic_coding/model_answer"
trajectories_dir = "agentic_code_runner/data/trajectories"

# Set maximum number of workers for parallel processing
MAX_WORKERS = 20

SWEAGENT_ERROR_STATUSES = [
    'exit_total_execution_time',
    'exit_command_timeout',
    'exit_context',
    'exit_api',
    'exit_environment_error',
    'exit_error',
    'exit_format',
    'exit_cost',
    'Uncaught TIMEOUT'
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
    # "Operation 'python"
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

# Set to track which (model, run_id, question_id, error) tuples have been found
# This helps avoid duplicates between JSONL and trajectory files
found_errors = set()

# Function to process a single JSONL file
def process_jsonl_file(jsonl_file):
    local_model_pairs = []
    local_model_errors = []
    local_found_errors = set()
    
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
                    
                    # Check for error pattern in the original JSONL line
                    for error_string in ALL_ERROR_STRINGS:
                        if error_string in line:
                            error_found = error_string
                            # Create a unique identifier for this error
                            error_id = (model_name, run_id, question_id, error_found)
                            # print(f"Found {error_id}")
                            index = line.find(error_string)
                            # print(line[index - 20:index + len(error_string) + 20])
                            # Only add if not already found
                            if error_id not in local_found_errors:
                                local_found_errors.add(error_id)
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
                                    if error_id not in local_found_errors:
                                        local_found_errors.add(error_id)
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
    jsonl_files = [os.path.join(jsonl_dir, f"{get_model_config(model_name).display_name}.jsonl") for model_name in args.model]
    # Filter out non-existent files
    jsonl_files = [f for f in jsonl_files if os.path.exists(f)]
    print(f"Found {len(jsonl_files)} JSONL files to process for specified models: {', '.join(args.model)}")
else:
    # Process all JSONL files
    jsonl_files = glob.glob(os.path.join(jsonl_dir, "*.jsonl"))
    print(f"Found {len(jsonl_files)} JSONL files to process")

with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # Submit all JSONL files for processing
    future_to_file = {executor.submit(process_jsonl_file, jsonl_file): jsonl_file for jsonl_file in jsonl_files}
    
    # Process results as they complete
    for future in concurrent.futures.as_completed(future_to_file):
        jsonl_file = future_to_file[future]
        try:
            model_name, local_pairs, local_errors, local_found = future.result()
            model_pairs[model_name].extend(local_pairs)
            model_errors[model_name].extend(local_errors)
            found_errors.update(local_found)
        except Exception as exc:
            print(f"Processing {jsonl_file} generated an exception: {exc}")
            traceback.print_exc()

# for jsonl_file in jsonl_files:
#     model_name, local_pairs, local_errors, local_found = process_jsonl_file(jsonl_file)
#     model_pairs[model_name].extend(local_pairs)
#     model_errors[model_name].extend(local_errors)
#     found_errors.update(local_found)

# Check each (run_id, question_id) pair for syntax errors
# for model, pairs in model_pairs.items():
#     print(f"Checking {len(pairs)} questions for model {model}")
    
#     for run_id, question_id in pairs:
#         # Directly construct the path to the specific folder
#         folder_path = os.path.join(trajectories_dir, run_id, question_id)
        
#         if not os.path.exists(folder_path):
#             continue

#         traj_path = os.path.join(folder_path, question_id + '.traj')

#         if not os.path.exists(traj_path):
#             continue
        
#         try:
#             with open(traj_path, 'r', errors='ignore') as f:
#                 content = f.read()
#                 for error_string in ALL_ERROR_STRINGS:
#                     if error_string in content:
#                         error_found = error_string
#                         # Check if this error was already found in the JSONL file
#                         error_id = (model, run_id, question_id, error_found)
#                         # Only add if not already found
#                         if error_id not in found_errors:
#                             found_errors.add(error_id)
#                             model_errors[model].append((run_id, question_id, error_found))
#         except Exception as e:
#             print(f"Error reading {traj_path}: {e}")
            

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
    for run_id, question_id, error in error_pairs:
        # Prefix to indicate where the error was found
        if error.startswith("JSONL:"):
            source = "JSONL file"
            error_type = error[6:]  # Remove the "JSONL:" prefix
        else:
            source = "Trajectory file"
            error_type = error
            
        run_id_to_errors[run_id][(source, error_type)].append(question_id)
    
    for run_id, error_to_question_ids in run_id_to_errors.items():
        print(f"  Run ID: {run_id}")
        overlap = None
        for (source, error), question_ids in error_to_question_ids.items():
            print(f"    {error}: {' '.join(question_ids)}")
            if overlap is None:
                overlap = set(question_ids)
            else:
                overlap = overlap.intersection(set(question_ids))
        
        if len(error_to_question_ids) > 1 and overlap is not None and len(overlap) > 0:
            print(f"  Overlapping question IDs: {' '.join(overlap)}")

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
            "--bench-name", "live_bench/coding/agentic_coding",
            "--mode", "sequential",
            "--question-source", "jsonl",
            "--parallel-requests", "20",
            "--parallel-grading", "20",
            "--question-id"] + list(question_ids)
        
        print(f"Running command: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, text=True)
            print(f"Command completed successfully for model {model}")
        except subprocess.CalledProcessError as e:
            print(f"Error running command for model {model}: {e}")
            print(f"Error output: {e.stderr}")