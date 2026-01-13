import re
import json


def extract_solution_from_response(response):
    """Extract solution from LLM response between <solution> tags."""
    # Extract from <solution>...</solution> tags (case-insensitive, take last match)
    solution_matches = re.findall(r'<solution>(.*?)</solution>', response, re.DOTALL | re.IGNORECASE)
    
    if len(solution_matches) == 0:
        return None
    
    content = solution_matches[-1].strip()
    
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def consecutive_events_process_results(ground_truth, llm_answer, debug=False):
    """
    Evaluate LLM answer for consecutive_events task using Jaccard similarity.
    
    Jaccard = |intersection| / |union| = TP / (TP + FP + FN)
    
    Args:
        ground_truth: List of dicts with user_id and max_consecutive_growth_months
        llm_answer: Raw LLM response string
        debug: Whether to print debug information
    
    Returns:
        float: Jaccard similarity score between 0.0 and 1.0
    """
    # Parse ground truth if it's a string
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except json.JSONDecodeError:
            if debug:
                print("Could not parse ground_truth as JSON")
            return 0.0
    
    # Extract solution from LLM response
    predicted = extract_solution_from_response(llm_answer)
    
    if predicted is None:
        if debug:
            print("Could not extract solution from LLM response")
            print("END OF OUTPUT:", llm_answer[-min(500, len(llm_answer)):])
        return 0.0
    
    # Handle empty cases
    if not ground_truth and not predicted:
        return 1.0
    
    if not ground_truth:
        if debug:
            print("Ground truth is empty but prediction is not")
        return 0.0
    
    if not predicted:
        if debug:
            print("Prediction is empty but ground truth is not")
        return 0.0
    
    # Build lookup dictionaries
    truth_dict = {r['user_id']: r['max_consecutive_growth_months'] for r in ground_truth}
    pred_dict = {}
    if predicted:
        for r in predicted:
            if isinstance(r, dict) and 'user_id' in r:
                pred_dict[r['user_id']] = r.get('max_consecutive_growth_months')
    
    # Calculate true positives (exact matches on both key and value)
    true_positives = sum(
        1 for user_id, expected_value in truth_dict.items()
        if user_id in pred_dict and pred_dict[user_id] == expected_value
    )
    
    # Calculate false positives and false negatives
    false_positives = len([uid for uid in pred_dict if uid not in truth_dict or pred_dict[uid] != truth_dict.get(uid)])
    false_negatives = len([uid for uid in truth_dict if uid not in pred_dict or pred_dict.get(uid) != truth_dict[uid]])
    
    # Jaccard = TP / (TP + FP + FN)
    union = true_positives + false_positives + false_negatives
    jaccard = true_positives / union if union > 0 else 0.0
    
    if debug:
        precision = true_positives / len(pred_dict) if len(pred_dict) > 0 else 0.0
        recall = true_positives / len(truth_dict) if len(truth_dict) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"True Positives: {true_positives}")
        print(f"False Positives: {false_positives}")
        print(f"False Negatives: {false_negatives}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Jaccard: {jaccard:.4f}")
        
        if jaccard < 1.0:
            print("GROUND TRUTH:", ground_truth)
            print("PREDICTED:", predicted)
    
    return round(jaccard, 4)