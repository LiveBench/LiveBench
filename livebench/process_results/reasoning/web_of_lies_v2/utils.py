
import re


def web_of_lies_process_results(ground_truth: str, llm_answer: str, debug=False) -> int:

    # pull out words in bold
    bold_words = re.findall(r'\*\*(.*?)\*\*', llm_answer)

    if not bold_words:
        if debug:
            print("NO BOLDS, answer", ground_truth, "output", llm_answer[-50:], )
        return 0

    last_bold = bold_words[-1].lower()

    # Check if last_bold is an exact match of ground_truth
    if last_bold == ground_truth.lower():
        return 1

    # Check if last_bold contains the ground_truth
    if last_bold.count("yes") + last_bold.count("no") == 3 and ground_truth.lower() in last_bold:
        return 1        

    if debug:
        print('FAILED, answer', ground_truth, 'output', last_bold)

    return 0
