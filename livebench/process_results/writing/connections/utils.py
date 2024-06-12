import re


def group_words(s):
    groups = [[]]
    words = s.split(",")
    words = [w.strip().lower() for w in words]
    for word in words:
        if len(groups[-1]) == 4:
            groups.append([])
        groups[-1].append(word)
    return groups


def connections_process_results(ground_truth: str, llm_answer: str) -> int:

    # pull out words in bold
    bold_words = re.findall(r'\*\*(.*?)\*\*', llm_answer)

    if not bold_words:
        return 0

    ground_truth_groups = group_words(ground_truth)
    max_score = 0
    for output_groups in list(map(group_words, bold_words)):

        correct_groups = 0
        for ground_truth_group in ground_truth_groups:
            for output_group in output_groups:
                if all([word in output_group for word in ground_truth_group]):
                    correct_groups += 1
                    break

        max_score = max(max_score, correct_groups / len(ground_truth_groups))
    return max_score
