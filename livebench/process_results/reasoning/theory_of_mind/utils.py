import re


def theory_of_mind_process_results(ground_truth: str, llm_answer: str, debug=False) -> int:
    """
    Process Theory of Mind task results by extracting answer from <solution> tags
    and comparing to ground truth.
    
    Args:
        ground_truth: The correct answer (e.g., "blue_container")
        llm_answer: The model's response
        debug: Whether to print debug information
        
    Returns:
        1 if correct, 0 if incorrect
    """
    # Extract text from <solution></solution> tags
    solution_matches = re.findall(r'<solution>(.*?)</solution>', llm_answer, re.IGNORECASE | re.DOTALL)
    
    if len(solution_matches) == 0:
        # Try to handle malformed tags
        solution_matches = re.findall(r'</solution>(.*?)</solution>', llm_answer, re.IGNORECASE | re.DOTALL)
    
    if len(solution_matches) == 0:
        # Fallback: try to find the answer in the last line
        last_line = llm_answer.strip().split('\n')[-1].strip()
        # Remove common prefixes
        for prefix in ['Answer:', 'answer:', 'The answer is', 'the answer is']:
            if last_line.startswith(prefix):
                last_line = last_line[len(prefix):].strip()
        solution_matches.append(last_line)
    
    if len(solution_matches) == 0:
        if debug:
            print('No solution text found for theory of mind')
            print('GROUND TRUTH:', ground_truth)
            print('END OF OUTPUT:', llm_answer[-100:])
        return 0
    
    # Use the last match (in case there are multiple)
    llm_solution = solution_matches[-1].strip().lower()
    
    # Clean up the solution text
    # Remove punctuation and extra whitespace
    llm_solution = re.sub(r'[.,!?;:]', '', llm_solution).strip()
    
    # Normalize ground truth
    gt_normalized = ground_truth.strip().lower()
    
    # Check if answers match
    score = 0
    if llm_solution == gt_normalized or gt_normalized in llm_solution:
        score = 1
    
    if debug and score == 0:
        print('INCORRECT')
        print('GROUND TRUTH:', gt_normalized)
        print('LLM SOLUTION:', llm_solution)
        print('END OF OUTPUT:', llm_answer[-100:])
    
    return score

