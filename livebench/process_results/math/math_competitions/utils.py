

def mathcontest_process_results(ground_truth: str, llm_answer: str) -> int:
    score = 0
    # the reference answer must be a single capital letter from A to E (I.e., the multiple choice answer)
    if not (isinstance(ground_truth, str) and len(ground_truth) == 1 and 'A' <= ground_truth <= 'E'):
        raise ValueError("amc_answer must be a single capital letter between A and E.")

    # The LLM was prompted to repeat letter answer 5 times, to make it easy to pull out the answer        
    if ground_truth * 4 in llm_answer:
        score = 1
    
    return score

def aime_process_results(ground_truth: str, llm_answer: str) -> int:
    score = 0
    if ground_truth in llm_answer[-20:]:
        score = 1
    
    return score

