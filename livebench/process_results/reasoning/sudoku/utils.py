"""
Sudoku puzzle evaluation utilities.

This module provides functions to evaluate Sudoku puzzle solutions by comparing
model answers against ground truth solutions.
"""

import re


def extract_solution_from_tags(text: str) -> str:
    """
    Extract solution from <solution> tags in the model's answer.

    Args:
        text: The model's full response text

    Returns:
        The extracted solution string, or empty string if no solution found
    """
    # Look for <solution>...</solution> tags (case insensitive)
    pattern = r'<solution>\s*(.*?)\s*</solution>'
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)

    if match:
        solution = match.group(1).strip()
        # Remove all whitespace (spaces, newlines, tabs) from the solution
        solution = re.sub(r'\s+', '', solution)
        return solution

    return ""


def normalize_solution(solution: str) -> str:
    """
    Normalize a solution string by removing whitespace.

    Keeps all non-whitespace characters including digits and dots (.).
    Dots represent empty cells in puzzles with extra cells outside the main board.

    Args:
        solution: The raw solution string

    Returns:
        Normalized solution with whitespace removed
    """
    # Remove all whitespace (spaces, tabs, newlines)
    normalized = re.sub(r'\s+', '', solution)
    return normalized


def sudoku_process_results(ground_truth: str, llm_answer: str, debug: bool = False) -> float:
    """
    Evaluate a Sudoku solution by comparing it to the ground truth.

    Args:
        ground_truth: The correct solution as a string of digits
        llm_answer: The model's full response text
        debug: Whether to print debug information

    Returns:
        1.0 if the solution is correct, 0.0 otherwise
    """
    # Extract solution from the model's answer
    extracted_solution = extract_solution_from_tags(llm_answer)

    # Normalize both solutions
    normalized_ground_truth = normalize_solution(ground_truth)
    normalized_llm_solution = normalize_solution(extracted_solution)

    # Check if lengths match first
    if len(normalized_llm_solution) != len(normalized_ground_truth):
        if debug:
            print(f"Length mismatch: expected {len(normalized_ground_truth)}, got {len(normalized_llm_solution)}")
        return 0.0

    # Exact match comparison
    if normalized_llm_solution == normalized_ground_truth:
        return 1.0
    else:
        if debug:
            print("Solution does not match")
            print(f"Normalized ground truth: {normalized_ground_truth}")
            print(f"Normalized LLM solution: {normalized_llm_solution}")
            # Show first difference
            for i, (c1, c2) in enumerate(zip(normalized_ground_truth, normalized_llm_solution)):
                if c1 != c2:
                    print(f"First difference at position {i}: expected '{c1}', got '{c2}'")
                    break
        return 0.0
