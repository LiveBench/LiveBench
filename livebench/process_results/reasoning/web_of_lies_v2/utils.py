
import re


def web_of_lies_process_results(ground_truth: str, llm_answer: str) -> int:

    # pull out words in bold
    bold_words = re.findall(r'\*\*(.*?)\*\*', llm_answer)

    if not bold_words:
        return 0

    last_bold = bold_words[-1].lower()

    # Check if last_bold is an exact match of ground_truth
    if last_bold == ground_truth.lower():
        return 1

    # Check if last_bold contains the ground_truth
    if last_bold.count("yes") + last_bold.count("no") == 3 and ground_truth.lower() in last_bold:
        return 1        

    return 0
