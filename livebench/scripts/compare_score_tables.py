#!/usr/bin/env python3

import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path

def compare_model_scores(file1_path, file2_path, threshold=0.001):
    """
    Compare model scores between two CSV files and print the differences.
    
    Args:
        file1_path: Path to first CSV file
        file2_path: Path to second CSV file
        threshold: Minimum difference to consider (to handle floating point differences)
    """
    # Verify files exist
    if not Path(file1_path).exists():
        print(f"Error: File not found: {file1_path}")
        sys.exit(1)
    if not Path(file2_path).exists():
        print(f"Error: File not found: {file2_path}")
        sys.exit(1)

    try:
        # Read the CSV files
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)
    except Exception as e:
        print(f"Error reading CSV files: {str(e)}")
        sys.exit(1)

    # Set 'model' as index for both dataframes
    df1.set_index('model', inplace=True)
    df2.set_index('model', inplace=True)
    
    # Get common models
    common_models = set(df1.index) & set(df2.index)
    
    # Get common columns (tasks)
    common_tasks = set(df1.columns) & set(df2.columns)
    
    print(f"Comparing {file1_path} vs {file2_path}\n")
    
    differences_found = False
    
    # For each common model
    for model in sorted(common_models):
        model_differences = []
        
        # Compare scores for each task
        for task in sorted(common_tasks):
            score1 = df1.loc[model, task]
            score2 = df2.loc[model, task]
            
            # Check if difference is greater than threshold
            if abs(score1 - score2) > threshold:
                model_differences.append(
                    f"  {task}: {score1:.3f} vs {score2:.3f} (diff: {score2 - score1:+.3f})"
                )
        
        # If there are differences for this model, print them
        if model_differences:
            differences_found = True
            print(f"Model: {model}")
            print("\n".join(model_differences))
            print()
    
    # Print models that are in one file but not the other
    only_in_file1 = set(df1.index) - set(df2.index)
    only_in_file2 = set(df2.index) - set(df1.index)
    
    if only_in_file1:
        print(f"Models only in {file1_path}:")
        print("\n".join(f"  {model}" for model in sorted(only_in_file1)))
        print()
        
    if only_in_file2:
        print(f"Models only in {file2_path}:")
        print("\n".join(f"  {model}" for model in sorted(only_in_file2)))
        print()
        
    if not differences_found and not only_in_file1 and not only_in_file2:
        print("No differences found between the files.")

def main():
    parser = argparse.ArgumentParser(description='Compare model scores between two CSV files')
    parser.add_argument('file1', help='Path to first CSV file')
    parser.add_argument('file2', help='Path to second CSV file')
    parser.add_argument('--threshold', type=float, default=0.001,
                      help='Minimum difference to consider (default: 0.001)')

    args = parser.parse_args()
    
    compare_model_scores(args.file1, args.file2, args.threshold)

if __name__ == "__main__":
    main()