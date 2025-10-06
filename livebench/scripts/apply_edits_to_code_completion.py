"""Apply edits from BCB generation to BCB completion."""

import json
import hashlib
import math
import numpy as np
from typing_extensions import Literal


def truncate_partial_solution(
    solution: str, difficulty: Literal["easy", "medium", "hard"]
):
    """
    Copy of the function from livebench-private/question_generation/coding/util.py.
    Used to generate partial solutions with the specified difficulty.
    """
    if difficulty == "easy":
        ratio = float(np.random.uniform(0.3, 0.7))
    else:
        ratio = 0.85

    num_lines = math.floor(len(solution.split("\n")) * ratio)
    partial_solution = "\n".join(solution.split("\n")[:num_lines])
    remainder = "\n".join(solution.split("\n")[num_lines:])

    assert solution == partial_solution + '\n' + remainder

    return partial_solution, remainder


def get_generic_question_template_answer_completion(question_content, partial_completion):
    """
    Copy of the function from livebench-private/question_generation/coding/util.py.
    Used to generate completion prompts with the specified partial completion.
    """
    prompt = "### Instructions: You are an expert Python programmer. You will be given a question (problem specification) and the first lines of Python solution to this problem, and will write in Python the remaining lines of the program to produce a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the second part of the program that you wrote.\n"
    prompt += f"### Question:\n{question_content}\n\n"

    prompt += "### Format: You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n"
    prompt += f"```python\n{partial_completion}\n```\n\n"

    prompt += f"### Answer: (enclose your partial completion in backticks. Only write the missing portion of the code, not the entire code. Be very careful to match the appropriate indentation. Directly appending your code after the partial code should produce a correct completion solution.)\n\n"
    return prompt


def generate_question_id(path_string, index):
    """
    Generate a question_id using the same method as in merge_questions.py.
    """
    prehash = f"{path_string}/{index}-2025-04-25"
    return hashlib.sha256(str.encode(prehash)).hexdigest()


def load_jsonl(file_path):
    """Load questions from a JSONL file."""
    questions = []
    with open(file_path, 'r') as f:
        for line in f:
            questions.append(json.loads(line))
    return questions


def save_jsonl(file_path, questions):
    """Save questions to a JSONL file."""
    with open(file_path, 'w') as f:
        for question in questions:
            f.write(json.dumps(question) + '\n')


def main():
    # Paths to the input and output files
    code_gen_path = "data/live_bench/coding/code_generation/question.jsonl"
    code_gen_original_path = "data/live_bench/coding/code_generation/question_edited_18.jsonl"
    code_comp_path = "data/live_bench/coding/code_completion/question.jsonl"
    output_path = "data/live_bench/coding/code_completion/question_edited.jsonl"
    
    # Path string for generating question IDs
    path_string = "live_bench/coding/code_completion_edited_3"
    
    # Load all questions
    code_gen_questions = load_jsonl(code_gen_path)
    code_gen_original_questions = load_jsonl(code_gen_original_path)
    code_comp_questions = load_jsonl(code_comp_path)
    
    # Create a mapping for easy lookup
    gen_questions_by_title = {q["question_title"]: q for q in code_gen_questions}
    gen_original_questions_by_title = {q["question_title"]: q for q in code_gen_original_questions}
    
    # Track what's changed to understand the edits
    changes_summary = {"tests": 0, "prompt": 0, "ground_truth_valid": 0, "ground_truth_regenerate": 0, "no_changes": 0}
    updated_questions = []

    for i, comp_question in enumerate(code_comp_questions):
        question_title = comp_question["question_title"]
        
        # Skip if this question doesn't exist in the generation set
        if question_title not in gen_questions_by_title or question_title not in gen_original_questions_by_title:
            print(f"Warning: Question title {question_title} not found in generation dataset. Skipping.")
            continue
        
        gen_question = gen_questions_by_title[question_title]
        gen_original_question = gen_original_questions_by_title[question_title]
        
        # Create a copy of the completion question to modify
        updated_question = comp_question.copy()
        need_to_regenerate_id = False
        changes_made = False
        
        # Check if tests have changed
        if gen_question["tests"] != gen_original_question["tests"]:
            updated_question["tests"] = gen_question["tests"]
            changes_made = True
            changes_summary["tests"] += 1
        
        # Check if prompt has changed
        if gen_question["turns"][0] != gen_original_question["turns"][0]:
            updated_question["turns"][0] = gen_question["turns"][0]
            need_to_regenerate_id = True
            changes_made = True
            changes_summary["prompt"] += 1
        
        # Check if ground truth has changed
        if gen_question["ground_truth"] != gen_original_question["ground_truth"]:
            # Check if the partial solution is still valid
            if gen_question["ground_truth"].startswith(comp_question["partial_solution"]):
                # Update ground_truth and remainder
                updated_question["ground_truth"] = gen_question["ground_truth"]
                updated_question["remainder"] = gen_question["ground_truth"][len(comp_question["partial_solution"]):]
                changes_made = True
                changes_summary["ground_truth_valid"] += 1
            else:
                # Need to regenerate partial solution
                np.random.seed(42)  # For reproducibility
                partial_solution, remainder = truncate_partial_solution(
                    gen_question["ground_truth"], difficulty="easy"
                )
                
                # Extract the description from the generation prompt
                description = gen_question["turns"][0].split("### Question:\n")[1].split("\n\n### Format:")[0]
                
                # Get the code_prompt from the generation question
                code_prompt = gen_question["code_prompt"]
                
                # Create the full partial solution (code_prompt + partial_solution)
                full_partial_solution = code_prompt + '\n' + partial_solution
                
                # Generate the updated completion prompt
                updated_prompt = get_generic_question_template_answer_completion(description, full_partial_solution)
                
                updated_question["ground_truth"] = gen_question["ground_truth"]
                updated_question["partial_solution"] = full_partial_solution
                updated_question["remainder"] = remainder
                updated_question["turns"][0] = updated_prompt
                
                need_to_regenerate_id = True
                changes_made = True
                changes_summary["ground_truth_regenerate"] += 1
            
            assert updated_question['partial_solution'] + updated_question['remainder'] == gen_question['ground_truth'] == updated_question['ground_truth']
        
        # Regenerate question_id if necessary
        if need_to_regenerate_id:
            updated_question["question_id"] = generate_question_id(path_string, i)
        
        if not changes_made:
            changes_summary["no_changes"] += 1
        else:
            print(f"Question {updated_question['question_id']} has changes.")
            
        updated_questions.append(updated_question)
    
    # Save the updated questions
    save_jsonl(output_path, updated_questions)
    
    # Print summary
    print(f"Updates complete. Changes summary:")
    print(f"  Tests: {changes_summary['tests']}")
    print(f"  Prompt updated: {changes_summary['prompt']}")
    print(f"  Ground truth with valid partial solution: {changes_summary['ground_truth_valid']}")
    print(f"  Ground truth with regenerated partial solution: {changes_summary['ground_truth_regenerate']}")
    print(f"  No changes: {changes_summary['no_changes']}")
    print(f"  Total questions processed: {len(code_comp_questions)}")
    print(f"  Total questions saved: {len(updated_questions)}")
    print(f"Updated questions saved to {output_path}")


if __name__ == "__main__":
    main()
