import json
import argparse
from pathlib import Path
import subprocess
import sys
from collections import defaultdict
import os
from livebench.model.api_model_config import get_model_config

def find_error_questions(root_dir, target_model_id=None, old_max_tokens=None):
    model_errors = defaultdict(list)

    model_display_name = get_model_config(target_model_id).display_name

    for jsonl_file in Path(root_dir).rglob('*.jsonl'):
        name = get_model_config(jsonl_file.stem).display_name
        if name != model_display_name:
            continue

        with open(jsonl_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    model_id = entry.get('model_id')

                    if target_model_id and model_id != target_model_id:
                        continue

                    choices = entry.get('choices', [])
                    if choices and isinstance(choices, list) and len(choices) > 0:
                        turns = choices[0].get('turns', [])
                        if turns and isinstance(turns, list) and '$ERROR$' in turns or len(turns) == 0 or turns[0] == '' or turns[0] == '<think>':
                            model_errors[model_id].append(entry['question_id'])
                        elif old_max_tokens and entry.get('total_output_tokens') >= old_max_tokens:
                            model_errors[model_id].append(entry['question_id'])
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON line in {jsonl_file}")
                except KeyError as e:
                    print(f"Warning: Missing required field {e} in {jsonl_file}")

    return model_errors

def run_commands_for_model(model_id, question_ids, new_max_tokens=None, api_base=None, api_key_name=None, mode=None, parallel_requests=None):
    if not question_ids:
        return

    model_name = model_id.split('/')[-1]

    # Build command for run_livebench.py
    cmd = ['python', 'run_livebench.py', '--model', model_name, 
           '--question-source', 'jsonl', '--question-id']
    cmd.extend(question_ids)

    if new_max_tokens:
        cmd.extend(['--max-tokens', str(new_max_tokens)])
    
    if api_base:
        cmd.extend(['--api-base', api_base])

    if api_key_name:
        cmd.extend(['--api-key-name', api_key_name])
        
    if mode:
        cmd.extend(['--mode', mode])
        
    if parallel_requests:
        cmd.extend(['--parallel-requests', str(parallel_requests)])

    print(f"\nProcessing model: {model_id}")
    print("\nExecuting run_livebench.py command:")
    print(' '.join(cmd))
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running run_livebench.py for {model_id}: {e}")
    except FileNotFoundError:
        print("Error: run_livebench.py not found")

def main():
    parser = argparse.ArgumentParser(description='Rerun failed questions.')
    parser.add_argument('--model-id', help='Target model ID')
    parser.add_argument('--old-max-tokens', type=int, default=None, help='Rerun questions for which there was no error but max tokens was exceeded')
    parser.add_argument('--new-max-tokens', type=int, help='Maximum number of tokens for the new run')
    parser.add_argument('--api-base', type=str, help='API base URL')
    parser.add_argument('--api-key-name', type=str, default=None)
    parser.add_argument('--mode', type=str, choices=['single', 'parallel', 'sequential'], 
                       help='Execution mode for run_livebench.py: single benchmark, parallel benchmarks, or sequential benchmarks')
    parser.add_argument('--parallel-requests', type=int, help='Number of parallel requests for API calls')
    
    args = parser.parse_args()
    
    root_dir = 'data'
    target_model_id = args.model_id
    if target_model_id:
        print(f"Target model ID: {target_model_id}")
    new_max_tokens = args.new_max_tokens
    if new_max_tokens:
        print(f"New max tokens: {new_max_tokens}")
    old_max_tokens = args.old_max_tokens
    if old_max_tokens is not None:
        print(f"Rerunning questions for which there was no error but max tokens was exceeded: {old_max_tokens} tokens")
    api_base = args.api_base
    if api_base:
        print(f"API base: {api_base}")
    api_key_name = args.api_key_name
    if api_key_name:
        print(f"API key name: {api_key_name}")
    mode = args.mode
    if mode:
        print(f"Mode: {mode}")
    parallel_requests = args.parallel_requests
    if parallel_requests:
        print(f"Parallel requests: {parallel_requests}")

    # Find all question IDs with errors
    model_errors = find_error_questions(root_dir, target_model_id, old_max_tokens)

    if not model_errors:
        print("No errors found!")
        return

    # Print results and run commands for each model
    for model_id, error_ids in model_errors.items():
        print(f"\nModel: {model_id}")
        print("Question IDs with $ERROR$ or max tokens exceeded:")
        print(' '.join(error_ids))

        # Run the livebench script for this model and failed questions
        run_commands_for_model(model_id, error_ids, new_max_tokens, api_base, api_key_name, mode, parallel_requests)

if __name__ == "__main__":
    main()