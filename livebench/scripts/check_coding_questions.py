from livebench.common import load_questions_jsonl, load_test_cases_jsonl
from livebench.process_results.coding.utils import LCB_generation_process_results

def check_coding_questions(questions_file):
    questions = load_questions_jsonl(questions_file)
    questions = load_test_cases_jsonl(questions_file, questions)

    for question in questions:
        full_solution = '```python\n' + question['remainder'] + '\n```'
        res = LCB_generation_process_results(question, full_solution, True)
        print(question['question_id'] + ': ' + str(res))

questions_file = 'data/live_bench/coding_2/coding_completion/question.jsonl'

check_coding_questions(questions_file)