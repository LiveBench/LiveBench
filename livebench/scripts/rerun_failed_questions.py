import json
import argparse
from pathlib import Path
import subprocess
import sys
from collections import defaultdict
import os
from livebench.model.api_model_config import get_model_config

def find_error_questions(root_dir, target_model_id=None, old_max_tokens=None, old_providers=None, replace_provider=None, bench_name=None, replace_api_name=None):
    model_errors = defaultdict(list)

    model_display_name = get_model_config(target_model_id).display_name

    print(f"Model display name: {model_display_name}")

    from_files = []
    total_output_tokens = 0

    for jsonl_file in Path(root_dir).rglob('*.jsonl'):
        if bench_name and bench_name not in str(jsonl_file.resolve()):
            continue

        name = get_model_config(jsonl_file.stem).display_name
        if name != model_display_name:
            continue

        added_questions_for_file = False

        with open(jsonl_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    model_id = entry.get('model_id')

                    if target_model_id and model_id != model_display_name:
                        continue

                    choices = entry.get('choices', [])
                    if choices and isinstance(choices, list) and len(choices) > 0:
                        turns = choices[0].get('turns', [])
                        if turns and isinstance(turns, list) and '$ERROR$' in turns or len(turns) == 0 or turns[0] == '' or turns[0] == '<think>':
                            model_errors[model_id].append(entry['question_id'])
                            added_questions_for_file = True
                            total_output_tokens += entry.get('total_output_tokens', 0)
                            continue
                    if old_max_tokens and entry.get('total_output_tokens') >= old_max_tokens:
                        model_errors[model_id].append(entry['question_id'])
                        added_questions_for_file = True
                        total_output_tokens += entry.get('total_output_tokens', 0)
                    elif old_providers and entry.get('api_info', {}).get('provider') in old_providers:
                        model_errors[model_id].append(entry['question_id'])
                        added_questions_for_file = True
                        total_output_tokens += entry.get('total_output_tokens', 0)
                    elif replace_provider and entry.get('api_info', {}).get('provider') != replace_provider:
                        model_errors[model_id].append(entry['question_id'])
                        added_questions_for_file = True
                        total_output_tokens += entry.get('total_output_tokens', 0)
                    elif replace_api_name and entry.get('api_info', {}).get('api_name') != replace_api_name:
                        model_errors[model_id].append(entry['question_id'])
                        added_questions_for_file = True
                        total_output_tokens += entry.get('total_output_tokens', 0)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON line in {jsonl_file}")
                except KeyError as e:
                    print(f"Warning: Missing required field {e} in {jsonl_file}")
        if added_questions_for_file:
            from_files.append(jsonl_file)

    for file in from_files:
        print(file)

    print(f"Total output tokens: {total_output_tokens}")

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
    parser.add_argument('--old-provider', type=str, nargs='+', help='Rerun questions for which this/these providers were used')
    parser.add_argument('--replace-provider', type=str, help='Replace the provider with this one')
    parser.add_argument('--bench-name', type=str, help='Benchmark name')
    parser.add_argument('--replace-api-name', type=str, help='Replace the API name with this one')
    
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
    old_providers = args.old_provider
    if old_providers:
        print(f"Old providers: {old_providers}")
    replace_provider = args.replace_provider
    if replace_provider:
        print(f"Replace provider: {replace_provider}")
    bench_name = args.bench_name
    if bench_name:
        print(f"Benchmark name: {bench_name}")
    replace_api_name = args.replace_api_name
    if replace_api_name:
        print(f"Replace API name: {replace_api_name}")
    # Find all question IDs with errors
    model_errors = find_error_questions(root_dir, target_model_id, old_max_tokens, old_providers, replace_provider, bench_name, replace_api_name)

    if not model_errors:
        print("No errors found!")
        return

    # Print results and run commands for each model
    for model_id, error_ids in model_errors.items():
        print(f"\nModel: {model_id}")
        print(f"{len(error_ids)} Question IDs to be rerun:")
        print(' '.join(error_ids))

        # Run the livebench script for this model and failed questions
        # run_commands_for_model(model_id, error_ids, new_max_tokens, api_base, api_key_name, mode, parallel_requests)

if __name__ == "__main__":
    main()