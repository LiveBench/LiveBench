from typing import Optional
import re


def mathcontest_process_results(ground_truth: str, llm_answer: str, question_text: str) -> int:
    score = 0
    # the reference answer must be a single capital letter from A to E (I.e., the multiple choice answer)
    if not (isinstance(ground_truth, str) and len(ground_truth) == 1 and 'A' <= ground_truth <= 'E'):
        raise ValueError("amc_answer must be a single capital letter between A and E.")

    # The LLM was prompted to repeat letter answer 5 times, to make it easy to pull out the answer        
    if ground_truth * 4 in llm_answer:
        score = 1

    allow_boxed = True
    if score == 0 and allow_boxed:
        llm_answer = llm_answer.replace("\\\\fbox{", "\\\\boxed{")
        last_boxed = last_boxed_only_string(llm_answer)
        if last_boxed:
            parsed_answer = remove_boxed(last_boxed)
            if parsed_answer == ground_truth:
                score = 1

    allow_answer_values = True
    if score == 0 and allow_answer_values:
        value = extract_answer(question_text, ground_truth)
        length_to_check = 20 + len(value)
        if value in llm_answer[-length_to_check:]:
            score = 1
    return score


def extract_answer(statement, letter):

    pattern = r'\\textbf{\(([A-E])\)\s?}(.*?)(?:\\qquad|\$)'
    matches = re.findall(pattern, statement)
    answers = {match[0]: match[1].strip() for match in matches}
    answer = answers.get(letter, None)

    if not answer or answer == "":
        answer = "FAILURE"

    answer = answer.strip()
    answer = answer.strip("$")
    answer = answer.strip("~")

    return answer


def aime_process_results(ground_truth: str, llm_answer: str) -> int:
    score = 0
    if ground_truth in llm_answer[-20:]:
        score = 1
    
    return score


def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1].replace("$", "").replace("fbox","boxed")

    return retval


def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]
