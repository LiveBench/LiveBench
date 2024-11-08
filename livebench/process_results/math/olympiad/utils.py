
def match_expression_completions_to_ground_truth(completions, ground_truth):
    num_matches = 0
    for i in range(len(ground_truth)):
        if i not in completions:
            continue

        completion = completions[i].lower().strip().replace(' ' , '')
        comp = ground_truth[i].lower().strip().replace(' ' , '')

        if completion == comp:
            num_matches += 1

    return num_matches/len(ground_truth)

def remove_nonnumeric_chars_at_ends(s):
    start_index = 0
    while start_index < len(s) and not s[start_index].isdigit():
        start_index += 1
    end_index = start_index
    while end_index < len(s) and s[end_index].isdigit():
        end_index += 1

    return s[start_index:end_index], len(s) - (end_index - start_index)

def extract_expression_completions_from_generation(generation):
    # generation has Answer: comma separated list of numbers. I want to extract the last such comma separated list
    split_string = "Answer"
    numbers = [k.strip() for k in generation.split(split_string)[-1].split(',')]

    # the last number may have some extra non-numeric characters at the end. Those need to be removed
    new_numbers = []
    for i, n in enumerate(numbers):
        n, num_removed = remove_nonnumeric_chars_at_ends(n)
        if n != '' and n != "₂":
            new_numbers.append(int(n))
        if (i > 0) and (num_removed > 0):
            break

    numbers = new_numbers
    return numbers

def proof_rearrangement_process_results(ground_truth: str, llm_answer: str, edit_distance=False) -> int:
    ground_truth = [int(n) for n in ground_truth.split(',')]

    completions = extract_expression_completions_from_generation(llm_answer)

    if edit_distance:
        from Levenshtein import distance
        match = distance(completions, ground_truth)
        frac_matches = 1-(match/max(len(completions), len(ground_truth)))
    else:
        match = [(completions[i] == ground_truth[i]) if i < len(ground_truth) else 0 for i in range(len(completions))]
        frac_matches = sum(match)/len(match) if len(match) > 0 else 0
        #print(ground_truth, completions, frac_matches)

    return frac_matches

