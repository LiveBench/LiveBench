#!/usr/bin/env python3
import json
import glob
import sys
import os

def find_max_attribute(file_prefix, attribute_name):
    # Find all matching .jsonl files recursively
    pattern = f"{file_prefix}.jsonl"
    max_value = float('-inf')
    count = 0
    max_value_question = None
    files_processed = 0

    print(f"Looking for files matching pattern: {pattern}")
    print(f"Starting from directory: {os.getcwd()}")

    # Walk through directory tree, following symlinks
    for root, dirs, files in os.walk('.', followlinks=True):
        print(f"Scanning directory: {root}")
        matching_files = [f for f in files if f == pattern]
        for filename in matching_files:
            filepath = os.path.join(root, filename)
            files_processed += 1
            print(f"Processing: {filepath}")

            # Read each line in the file
            with open(filepath, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        # Parse JSON object
                        obj = json.loads(line.strip())

                        # Check if attribute exists
                        if attribute_name not in obj:
                            print(f"Warning: {attribute_name} not found in object at line {line_num} in {filepath}")
                            continue

                        # Update max value
                        current_value = float(obj[attribute_name])
                        if current_value > max_value:
                            count = 1
                            max_value = current_value
                            max_value_question = obj
                        elif current_value == max_value:
                            count += 1
                        print(f"Found value: {current_value} ({obj['question_id']}) (current max: {max_value})")

                    except json.JSONDecodeError:
                        print(f"Warning: Invalid JSON at line {line_num} in {filepath}")
                    except ValueError:
                        print(f"Warning: Could not convert {attribute_name} to number at line {line_num} in {filepath}")

    if files_processed == 0:
        print(f"\nNo files matching '{pattern}' found")
        print("\nDirectory structure:")
        for root, dirs, files in os.walk('.', followlinks=True):
            print(f"\nDirectory: {root}")
            print("Files:", files)
            print("Subdirectories:", dirs)
        return None

    return max_value, count, max_value_question

def main():
    if len(sys.argv) != 3:
        print("Usage: script.py <file_prefix> <attribute_name>")
        print("Example: script.py data temperature")
        sys.exit(1)

    file_prefix = sys.argv[1]
    attribute_name = sys.argv[2]

    res = find_max_attribute(file_prefix, attribute_name)

    if res is not None:
        max_value, count, max_value_question = res
        print(f"\nMaximum value of '{attribute_name}' across all files: {max_value} ({count} times)")
        print(f"\nQuestion was {max_value_question['question_id']}")

if __name__ == "__main__":
    main()

# No files are created during execution