import re


def clean_text(text):
    text = text.lower().strip()    
    text = re.sub(r'[^\w]', '', text)
    return text


def cta_process_results(ground_truth: str, llm_answer: str) -> int:

    if clean_text(ground_truth) == clean_text(llm_answer):
        return 1
    else:
        return 0
