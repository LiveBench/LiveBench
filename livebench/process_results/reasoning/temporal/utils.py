import re
from livebench.process_results.util import last_boxed_only_string, remove_boxed

def temporal_process_results(ground_truth: str, llm_answer: str, debug=False) -> int:
    solution_matches = re.findall(r'<solution>(.*?)<\/solution>', llm_answer)

    if len(solution_matches) == 0:
        solution_matches = re.findall(r'</solution>(.*?)</solution>', llm_answer)
    
    if len(solution_matches) == 0 and ('\\boxed' in llm_answer or '\\fbox{' in llm_answer):
        llm_answer = llm_answer.replace("\\\\fbox{", "\\\\boxed{")
        last_boxed = last_boxed_only_string(llm_answer)
        if last_boxed:
            boxed_removed = remove_boxed(last_boxed)
            boxed_removed = boxed_removed.replace("\\text{", "").replace("}", "").replace('\\', '')
            solution_matches.append(boxed_removed)

    if len(solution_matches) == 0:
        if debug:
            print('No solution text found for temporal')
            print('GROUND TRUTH', ground_truth)
            print('END OF OUTPUT', llm_answer[-100:])
        return 0

    if solution_matches[-1] == ground_truth:
        return 1

    if debug:
        print('INCORRECT')
        print('GROUND TRUTH', ground_truth)
        print('SOLUTION', solution_matches[-1])
        print('END OF OUTPUT', llm_answer[-100:])
    return 0