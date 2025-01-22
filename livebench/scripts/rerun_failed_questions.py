import json
import argparse
from pathlib import Path
import subprocess
import sys
from collections import defaultdict

def find_error_questions(root_dir, target_model_id=None, include_max_tokens=None):
    model_errors = defaultdict(list)

    for jsonl_file in Path(root_dir).rglob('*.jsonl'):
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
                        if turns and isinstance(turns, list) and '$ERROR$' in turns or len(turns) == 0 or turns[0] == '':
                            model_errors[model_id].append(entry['question_id'])
                        elif include_max_tokens and entry.get('total_output_tokens') == include_max_tokens:
                            model_errors[model_id].append(entry['question_id'])
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON line in {jsonl_file}")
                except KeyError as e:
                    print(f"Warning: Missing required field {e} in {jsonl_file}")

    return model_errors

def run_commands_for_model(model_id, question_ids, max_tokens=None, api_base=None):
    if not question_ids:
        return

    model_name = model_id.split('/')[-1]

    # First command: gen_api_answer.py
    cmd1 = ['python', 'gen_api_answer.py', '--model', model_name, 
            '--question-source', 'jsonl', '--question-id'] + question_ids

    if max_tokens:
        cmd1.extend(['--max-tokens', str(max_tokens)])
    
    if api_base:
        cmd1.extend(['--api-base', api_base])

    # Second command: gen_ground_truth_judgment.py
    cmd2 = ['python', 'gen_ground_truth_judgment.py', '--model', model_id,
            '--question-source', 'jsonl', '--question-id'] + question_ids

    print(f"\nProcessing model: {model_id}")

    # Run gen_api_answer.py
    print("\nExecuting gen_api_answer command:")
    print(' '.join(cmd1))
    try:
        subprocess.run(cmd1, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running gen_api_answer.py for {model_id}: {e}")
        return  # Skip ground truth if api answer fails
    except FileNotFoundError:
        print("Error: gen_api_answer.py not found in current directory")
        return

    # Run gen_ground_truth_judgment.py
    print("\nExecuting gen_ground_truth_judgment command:")
    print(' '.join(cmd2))
    try:
        subprocess.run(cmd2, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running gen_ground_truth_judgment.py for {model_id}: {e}")
    except FileNotFoundError:
        print("Error: gen_ground_truth_judgment.py not found in current directory")

def main():
    parser = argparse.ArgumentParser(description='Rerun failed questions.')
    parser.add_argument('root_directory', help='Root directory path')
    parser.add_argument('--model-id', help='Target model ID')
    parser.add_argument('--rerun-token-error', type=int, default=None, help='Rerun questions for which there was no error but max tokens was exceeded')
    parser.add_argument('--max-tokens', type=int, help='Maximum number of tokens')
    parser.add_argument('--api-base', type=str, help='API base URL')
    
    args = parser.parse_args()
    
    root_dir = args.root_directory
    target_model_id = args.model_id
    if target_model_id:
        print(f"Target model ID: {target_model_id}")
    max_tokens = args.max_tokens
    if max_tokens:
        print(f"Max tokens: {max_tokens}")
    include_max_tokens = args.rerun_token_error
    if include_max_tokens is not None:
        print(f"Rerunning questions for which there was no error but max tokens was exceeded: {include_max_tokens} tokens")
    api_base = args.api_base
    if api_base:
        print(f"API base: {api_base}")

    # Find all question IDs with errors
    model_errors = find_error_questions(root_dir, target_model_id, include_max_tokens)

    if not model_errors:
        print("No errors found!")
        return

    # Print results and run commands for each model
    for model_id, error_ids in model_errors.items():
        print(f"\nModel: {model_id}")
        print("Question IDs with $ERROR$:")
        for qid in error_ids:
            print(qid)

        # Run both commands in sequence for this model
        run_commands_for_model(model_id, error_ids, max_tokens, api_base)

if __name__ == "__main__":
    main()