
import re


def web_of_lies_process_results(ground_truth: str, llm_answer: str) -> int:

    # pull out words in bold
    bold_words = re.findall(r'\*\*(.*?)\*\*', llm_answer)

    # Check if any bold words were found and compare the last one to the ground_truth
    if bold_words and bold_words[-1].lower() == ground_truth.lower():
        return 1
    else:
        return 0

def house_traversal_process_results(ground_truth: str, llm_answer: str) -> int:

    # pull out words in bold
    bold_words = re.findall(r'\*\*(.*?)\*\*', llm_answer.lower())
    if not len(bold_words):
        return 0

    last_bold = bold_words[-1]
    answers = last_bold.split(", ")
    gt_answers = ground_truth.lower().split(", ")

    if len(answers) < len(gt_answers):
        return 0
    if all([answers[i] == gt_answers[i] for i in range(len(gt_answers))]):
        return 1
    return 0
