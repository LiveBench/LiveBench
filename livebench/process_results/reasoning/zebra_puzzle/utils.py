
import re


def zebra_puzzle_process_results(ground_truth: str, llm_answer: str) -> int:
    # Mapping of numbers to words for 1 to 9
    number_to_word = {
        '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
    }

    # Pull out words in bold
    bold_words = re.findall(r'\*\*\*(\w+)\*\*\*', llm_answer)

    # Remove any trailing punctuation from the last bold word if exists
    if bold_words:
        if (bold_words[-1].lower() == ground_truth.lower() or
            (bold_words[-1] in number_to_word and number_to_word[bold_words[-1]].lower() == ground_truth.lower())
            or bold_words[-1].lower() + ' movies' == ground_truth.lower()):
            return 1
        else:
            return 0
    else:
        # Split the text into words and remove punctuation.
        words = re.findall(r'\b\w+\b', llm_answer)
        last_word = words[-1] if words else ''
        # Check if the last bold word is a number and matches the word representation of the ground_truth
        if (last_word.lower() == ground_truth.lower() or
            (last_word in number_to_word and number_to_word[last_word].lower() == ground_truth.lower())
            or last_word.lower() + ' movies' == ground_truth.lower()):
            return 1
        return 0
