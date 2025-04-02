import os
import json
from collections import defaultdict

def check_errors():
    """
    Check for errors in model answer files and display them in a nicely formatted way,
    grouped by model and file.
    """
    
    # Navigate to the data directory
    os.chdir('data')

    # Dictionary to store results organized by model and task
    results = defaultdict(lambda: defaultdict(list))
    
    # Walk through data directory structure
    for root, dirs, files in os.walk('live_bench'):
        for dir in dirs:
            if dir == 'model_answer':
                model_dir = os.path.join(root, dir)
                # Extract task name from the path
                task_path = os.path.relpath(model_dir, 'live_bench')
                task_name = task_path.split('/')[0]  # First part of the path is the task category
                task_subname = task_path.split('/')[1]  # Second part is specific task
                full_task = f"{task_name}/{task_subname}"
                print(f"Checking {full_task}")
                
                # Process all JSONL files in the model_answer directory
                for json_file in [f for f in os.listdir(model_dir) if f.endswith('.jsonl')]:
                    file_path = os.path.join(model_dir, json_file)
                    model_name = json_file.replace('.jsonl', '')
                    
                    # Read the file and check for errors
                    with open(file_path, 'r') as f:
                        for line_num, line in enumerate(f, 1):
                            try:
                                data = json.loads(line)
                                
                                # Check for "ERROR" in the content
                                if "ERROR" in str(data):
                                    question_id = data.get('question_id', 'Unknown')
                                    results[model_name][full_task].append({
                                        'line': line_num,
                                        'question_id': question_id,
                                        'type': 'ERROR',
                                        'file': file_path
                                    })
                                
                                # Check for empty turns
                                if 'choices' in data:
                                    for choice in data['choices']:
                                        if 'turns' in choice and choice['turns'] == [""]:
                                            question_id = data.get('question_id', 'Unknown')
                                            results[model_name][full_task].append({
                                                'line': line_num,
                                                'question_id': question_id,
                                                'type': 'EMPTY_TURNS',
                                                'file': file_path
                                            })
                            except json.JSONDecodeError:
                                # Skip malformed JSON
                                continue
    
    # Display results grouped by model and task
    if not results:
        print("No errors found.")
        return
    
    total_errors = 0
    total_empty_turns = 0
    
    print("\n" + "="*80)
    print(" ERROR CHECK RESULTS ".center(80, "="))
    print("="*80 + "\n")
    
    for model_name in sorted(results.keys()):
        model_error_count = 0
        model_empty_count = 0
        
        print(f"\n{model_name}:")
        print("-" * (len(model_name) + 1))
        
        for task, errors in sorted(results[model_name].items()):
            error_count = sum(1 for e in errors if e['type'] == 'ERROR')
            empty_count = sum(1 for e in errors if e['type'] == 'EMPTY_TURNS')
            
            if error_count > 0 or empty_count > 0:
                print(f"\n  Task: {task}")
                
                if error_count > 0:
                    print(f"    Error entries: {error_count}")
                    for e in [e for e in errors if e['type'] == 'ERROR']:
                        print(f"      Line {e['line']}: Question ID {e['question_id']}")
                
                if empty_count > 0:
                    print(f"    Empty turns: {empty_count}")
                    for e in [e for e in errors if e['type'] == 'EMPTY_TURNS']:
                        print(f"      Line {e['line']}: Question ID {e['question_id']}")
            
            model_error_count += error_count
            model_empty_count += empty_count
        
        print(f"\n  Total for {model_name}: {model_error_count} errors, {model_empty_count} empty turns")
        total_errors += model_error_count
        total_empty_turns += model_empty_count
    
    print("\n" + "="*80)
    print(f" SUMMARY: {total_errors} errors, {total_empty_turns} empty turns ".center(80, "="))
    print("="*80 + "\n")

if __name__ == "__main__":
    check_errors()