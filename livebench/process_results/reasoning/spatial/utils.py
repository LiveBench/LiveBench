
import re


def spatial_process_results(ground_truth: str, llm_answer: str) -> int:

    word_to_number = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
        'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
        'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14', 'fifteen': '15',
        'sixteen': '16', 'seventeen': '17', 'eighteen': '18', 'nineteen': '19', 'twenty': '20'
    }

    bold_words = re.findall(r'\*\*([^\*]+)\*\*', llm_answer)

    # allow the answer to be within the last 3 bolded words
    words_to_check = []
    for i in range(3):
        if bold_words and len(bold_words) > i:
            words_to_check.append(bold_words[-i-1].strip().lower())

    for i, word in enumerate(words_to_check):
        if word == ground_truth.strip().lower():
            return 1

        # allow the answer to be the number spelled out
        if word in word_to_number and word_to_number[word] == ground_truth.strip().lower():
            return 1

        # allow certain cases like "two tetrahedra" == "tetrahedra" and "equilateral triangle" == "triangle"
        # while still disallowing cases like "circle square triangle" == "triangle"
        for answer in ["tetrahedra", "tetrahedron", "triangle", "square"]:
            if ground_truth.strip().lower() == answer and answer in word and len(word) < (2 * len(answer) + 5):
                return 1

    debug = False
    if debug and bold_words:
        print('failed, answer vs ground_truth:', bold_words[-1].strip().lower(), ground_truth.strip().lower())
        print("failed, full bold:", bold_words)
    elif debug:
        print("failed, did not have bolds", llm_answer[-20:], "vs", ground_truth.strip().lower())
    return 0
