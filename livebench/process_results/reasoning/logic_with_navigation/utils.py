import re


def logic_with_navigation_process_results(ground_truth: list, llm_answer: str, debug=False) -> int:
    """
    Process results for logic_with_navigation task.

    Args:
        ground_truth: A list of two integers [x, y] representing the expected final position
        llm_answer: The model's answer string, expected to contain coordinates in <solution> tags
        debug: If True, print debug information for incorrect answers

    Returns:
        1 if correct, 0 if incorrect
    """
    score = 0
    parsed_answer = None

    # Extract text from <solution></solution> tags
    solution_matches = re.findall(r'<solution>(.*?)</solution>', llm_answer, re.DOTALL | re.IGNORECASE)

    if len(solution_matches) > 0:
        # Take the last solution tag
        solution_text = solution_matches[-1].strip()

        # Try to parse coordinates from various formats
        # Format 1: (x, y) or (x,y)
        coord_match = re.search(r'\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)', solution_text)
        if coord_match:
            try:
                x = int(coord_match.group(1))
                y = int(coord_match.group(2))
                parsed_answer = [x, y]
            except ValueError:
                parsed_answer = None

        # Format 2: x, y (without parentheses)
        if parsed_answer is None:
            coord_match = re.search(r'(-?\d+)\s*,\s*(-?\d+)', solution_text)
            if coord_match:
                try:
                    x = int(coord_match.group(1))
                    y = int(coord_match.group(2))
                    parsed_answer = [x, y]
                except ValueError:
                    parsed_answer = None

    # Check if parsed answer matches ground truth
    if parsed_answer is not None and len(parsed_answer) == 2:
        if parsed_answer[0] == ground_truth[0] and parsed_answer[1] == ground_truth[1]:
            score = 1

    if debug and score == 0:
        print("INCORRECT")
        print("GROUND TRUTH:", ground_truth)
        print("PARSED ANSWER:", parsed_answer)
        if solution_matches:
            print("SOLUTION TEXT:", solution_matches[-1][:100])
        print("END OF OUTPUT:", llm_answer[-100:])

    return score
