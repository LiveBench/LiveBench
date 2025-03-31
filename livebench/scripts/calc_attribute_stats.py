#!/usr/bin/env python3
import json
import glob
import sys
import os
import argparse

def calc_attribute_stats(file_prefixes, attribute_name):
    # Track stats by task
    stats_by_task = {}
    # Overall stats
    overall_stats = {
        'max_value': float('-inf'),
        'max_value_question': None,
        'max_count': 0,
        'total_value': 0,
        'value_count': 0
    }
    files_processed = 0

    if isinstance(file_prefixes, str):
        file_prefixes = [file_prefixes]  # Convert single prefix to list for consistency

    print(f"Looking for files matching prefixes: {file_prefixes}")
    print(f"Starting from directory: {os.getcwd()}")

    # Walk through directory tree, following symlinks
    for root, dirs, files in os.walk('.', followlinks=True):
        print(f"Scanning directory: {root}")
        
        # Check for files matching any of the provided prefixes
        matching_files = []
        for prefix in file_prefixes:
            pattern = f"{prefix}.jsonl"
            matching_files.extend([f for f in files if f == pattern])
        
        for filename in matching_files:
            filepath = os.path.join(root, filename)
            files_processed += 1
            print(f"Processing: {filepath}")
            
            # Extract task name (parent of parent folder)
            path_parts = os.path.normpath(root).split(os.sep)
            task_name = path_parts[-2] if len(path_parts) >= 3 else "unknown"
            
            # Initialize task stats if needed
            if task_name not in stats_by_task:
                stats_by_task[task_name] = {
                    'max_value': float('-inf'),
                    'max_value_question': None,
                    'max_count': 0,
                    'total_value': 0,
                    'value_count': 0
                }

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

                        # Get current value
                        current_value = float(obj[attribute_name])
                        
                        # Update task stats
                        task_stats = stats_by_task[task_name]
                        task_stats['total_value'] += current_value
                        task_stats['value_count'] += 1
                        
                        if current_value > task_stats['max_value']:
                            task_stats['max_count'] = 1
                            task_stats['max_value'] = current_value
                            task_stats['max_value_question'] = obj
                        elif current_value == task_stats['max_value']:
                            task_stats['max_count'] += 1
                            
                        # Update overall stats
                        overall_stats['total_value'] += current_value
                        overall_stats['value_count'] += 1
                        
                        if current_value > overall_stats['max_value']:
                            overall_stats['max_count'] = 1
                            overall_stats['max_value'] = current_value
                            overall_stats['max_value_question'] = obj
                        elif current_value == overall_stats['max_value']:
                            overall_stats['max_count'] += 1
                            
                        print(f"Found value: {current_value} ({obj['question_id']}) (current max: {overall_stats['max_value']})")

                    except json.JSONDecodeError:
                        print(f"Warning: Invalid JSON at line {line_num} in {filepath}")
                    except ValueError:
                        print(f"Warning: Could not convert {attribute_name} to number at line {line_num} in {filepath}")

    if files_processed == 0:
        print(f"\nNo files matching '{file_prefixes}' found")
        print("\nDirectory structure:")
        for root, dirs, files in os.walk('.', followlinks=True):
            print(f"\nDirectory: {root}")
            print("Files:", files)
            print("Subdirectories:", dirs)
        return None

    # Calculate averages for each task
    for task, stats in stats_by_task.items():
        if stats['value_count'] > 0:
            stats['average_value'] = stats['total_value'] / stats['value_count']
        else:
            stats['average_value'] = 0
    
    # Calculate overall average
    if overall_stats['value_count'] > 0:
        overall_stats['average_value'] = overall_stats['total_value'] / overall_stats['value_count']
    else:
        overall_stats['average_value'] = 0

    return stats_by_task, overall_stats

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Calculate statistics for an attribute in JSON files')
    parser.add_argument('--file-name', dest='file_prefixes', nargs='+', required=True,
                        help='File prefixes to search for')
    parser.add_argument('--attribute', dest='attribute_name', required=True,
                        help='Name of the attribute to analyze')
    parser.add_argument('--stat', choices=['max', 'avg', 'all'], default='all',
                      help='Which statistics to display: max, avg, or all (default: all)')
    
    args = parser.parse_args()
    
    print(f"File prefixes: {args.file_prefixes}")
    print(f"Attribute name: {args.attribute_name}")
    print(f"Statistics mode: {args.stat}")

    res = calc_attribute_stats(args.file_prefixes, args.attribute_name)

    if res is not None:
        stats_by_task, overall_stats = res
        
        # Print stats by task in table format
        print("\n===== Statistics by Task =====")
        
        # Define the table headers based on the stat argument
        if args.stat == 'max':
            headers = ["Task", f"Max {args.attribute_name}", "Count", "Max Question ID"]
        elif args.stat == 'avg':
            headers = ["Task", f"Avg {args.attribute_name}", "Total", "Samples"]
        else:  # 'all'
            headers = ["Task", f"Max {args.attribute_name}", "Count", f"Avg {args.attribute_name}", 
                      "Total", "Samples", "Max Question ID"]
        
        # Calculate column widths
        col_widths = [max(len(h), 20) for h in headers]
        
        # Print header row
        header_row = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        print(header_row)
        
        # Print separator
        separator = "-" * len(header_row)
        print(separator)
        
        # Print data rows
        for task, stats in sorted(stats_by_task.items(), key=lambda x: x[1]['total_value'], reverse=True):
            question_id = stats['max_value_question']['question_id'] if stats['max_value_question'] else "N/A"
            
            # Prepare row data based on stat argument
            if args.stat == 'max':
                row = [
                    task,
                    f"{stats['max_value']:.4f}",
                    str(stats['max_count']),
                    question_id
                ]
            elif args.stat == 'avg':
                row = [
                    task,
                    f"{stats['average_value']:.4f}",
                    f"{stats['total_value']:.4f}",
                    str(stats['value_count'])
                ]
            else:  # 'all'
                row = [
                    task,
                    f"{stats['max_value']:.4f}",
                    str(stats['max_count']),
                    f"{stats['average_value']:.4f}",
                    f"{stats['total_value']:.4f}",
                    str(stats['value_count']),
                    question_id
                ]
                
            formatted_row = " | ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths))
            print(formatted_row)
        
        # Print separator
        print(separator)
            
        # Print overall stats
        print("\n===== Overall Statistics =====")
        if args.stat in ['max', 'all']:
            print(f"Maximum value of '{args.attribute_name}' across all files: {overall_stats['max_value']} (appeared {overall_stats['max_count']} times)")
            if overall_stats['max_value_question']:
                print(f"Question with max value: {overall_stats['max_value_question']['question_id']}")
        
        if args.stat in ['avg', 'all']:
            print(f"Average value of '{args.attribute_name}' across all files: {overall_stats['average_value']:.4f}")
            print(f"Total value of '{args.attribute_name}' across all files: {overall_stats['total_value']}")
            print(f"Number of values across all files: {overall_stats['value_count']}")

if __name__ == "__main__":
    main()

# No files are created during execution