"""
Script to edit questions in LiveBench dataset.
"""

import argparse
import os
import glob
import json

from livebench.common import load_questions_jsonl, make_match_single
from livebench.gen_ground_truth_judgment import play_a_match_gt
from livebench.model import get_model_config


def prepare_edit(question_ids: list[str]):
    """
    Prepare questions for editing.
    
    Args:
        question_ids: List of question IDs to prepare for editing
        data_dir: Base directory for LiveBench data
    """
    # Find all question.jsonl files in data/live_bench/
    data_dir = "data"
    question_files = glob.glob(os.path.join(data_dir, "live_bench", "**", "question.jsonl"), recursive=True)
    
    for question_file in question_files:
        # Load questions from the file
        questions = load_questions_jsonl(question_file, question_ids=question_ids)
        
        # Extract question_file_path from the full path (removing data/ prefix)
        rel_path = os.path.relpath(question_file, data_dir)
        question_file_path = os.path.dirname(rel_path)
        
        # Process each matching question
        for question in questions:
            q_id = question["question_id"]
            
            # Create output directory
            output_dir = os.path.join("question_edit", question_file_path, str(q_id))
            os.makedirs(output_dir, exist_ok=True)
            
            # Extract prompt, ground truth, and tests
            prompt = question.get("turns", [""])[0]
            
            if "code_generation" in question_file_path or "code_completion" in question_file_path:
                ground_truth = question.get("ground_truth", "")
                code_prompt = question.get("code_prompt", "")
                if not ground_truth.startswith(code_prompt):
                    ground_truth = code_prompt + "\n" + ground_truth
                tests = question.get("tests", "")
            else:
                raise NotImplementedError(f"Question type {question_file_path} not supported")
            
            # Write files
            with open(os.path.join(output_dir, "prompt.md"), "w") as f:
                f.write(prompt)
                
            with open(os.path.join(output_dir, "ground_truth.py"), "w") as f:
                f.write(ground_truth)
                
            if tests:
                with open(os.path.join(output_dir, "tests.py"), "w") as f:
                    f.write(tests)
            
            print(f"Prepared question {q_id} from {question_file}")


def evaluate(question_id: str, model: str | None = None):
    """
    Evaluate an edited question using play_a_match_gt.
    
    Args:
        question_id: ID of the question to evaluate
        model: If provided, also evaluate this model's answer in addition to ground truth
    """
    # Find the edited question directory
    data_dir = "data"
    edit_dir = glob.glob(os.path.join("question_edit", "live_bench", "**", str(question_id)), recursive=True)
    
    if not edit_dir:
        raise ValueError(f"Could not find edited question with ID {question_id}")
    
    edit_dir = edit_dir[0]
    
    # Determine the relative path to the original question
    question_rel_path = os.path.relpath(os.path.dirname(edit_dir), "question_edit")
    question_file = os.path.join(data_dir, question_rel_path, "question.jsonl")
    
    if not os.path.exists(question_file):
        raise ValueError(f"Original question file not found: {question_file}")
    
    # Load the original question
    questions = load_questions_jsonl(question_file, question_ids=[question_id])
    
    if not questions:
        raise ValueError(f"Could not find question with ID {question_id} in {question_file}")
    
    question = questions[0]
    
    # Read edited files
    prompt_file = os.path.join(edit_dir, "prompt.md")
    ground_truth_file = os.path.join(edit_dir, "ground_truth.py")
    tests_file = os.path.join(edit_dir, "tests.py")
    
    if not os.path.exists(prompt_file) or not os.path.exists(ground_truth_file):
        raise ValueError(f"Missing required edited files in {edit_dir}")
    
    with open(prompt_file, "r") as f:
        prompt = f.read()
    
    with open(ground_truth_file, "r") as f:
        ground_truth = f.read()
    
    tests = ""
    if os.path.exists(tests_file):
        with open(tests_file, "r") as f:
            tests = f.read()
    
    question["turns"] = [prompt]
    # Use ground truth
    question["ground_truth"] = ground_truth
        
    if tests:
        question["tests"] = tests

    # Always run ground truth evaluation
    gt_match = make_match_single([question], ["ground_truth"], {
        "ground_truth": {question["question_id"]: {"choices": [{"turns": [question["ground_truth"]]}]}}
    })[0]
    
    try:
        gt_result = play_a_match_gt(gt_match, output_file=None, debug=True)
        print("Ground Truth Evaluation:")
        print(gt_result)
    except Exception as e:
        print(f"Error evaluating ground truth for question {question_id}: {str(e)}")
        gt_result = None

    # If model is provided, also run model evaluation
    if model:
        model_name = get_model_config(model).display_name
        
        # Load model answer
        model_answer_file = os.path.join(data_dir, question_rel_path, "model_answer", f"{model_name}.jsonl")
        
        if not os.path.exists(model_answer_file):
            raise ValueError(f"Model answer file not found: {model_answer_file}")
        
        # Load model answers and find the one matching our question ID
        with open(model_answer_file, "r") as f:
            model_answers = [json.loads(line) for line in f.readlines()]
        
        model_answer = None
        for answer in model_answers:
            if answer.get("question_id") == question_id:
                model_answer = answer
                break
        
        if not model_answer:
            raise ValueError(f"Could not find answer for question {question_id} in {model_answer_file}")
        
        # Create a match using model answer
        model_match = make_match_single([question], [model_name], {
            model_name: {question["question_id"]: model_answer}
        })[0]
        
        try:
            print(f"\nModel Evaluation ({model_name}):")
            model_result = play_a_match_gt(model_match, output_file=None, debug=True)
            print(model_result)
            return {"ground_truth": gt_result, "model": model_result}
        except Exception as e:
            print(f"Error evaluating model {model_name} for question {question_id}: {str(e)}")
            return {"ground_truth": gt_result, "model": None}
    
    return {"ground_truth": gt_result}


def save_edit(question_ids: list[str]):
    """
    Save edited questions back to the LiveBench dataset.
    
    Args:
        question_ids: List of question IDs to save
    """
    # Find all edited question directories for the specified question IDs
    data_dir = "data"
    for question_id in question_ids:
        edit_dirs = glob.glob(os.path.join("question_edit", "live_bench", "**", str(question_id)), recursive=True)
        
        if not edit_dirs:
            print(f"Could not find edited question with ID {question_id}, skipping")
            continue
        
        edit_dir = edit_dirs[0]
        
        # Determine the relative path to the original question
        question_rel_path = os.path.relpath(os.path.dirname(edit_dir), "question_edit")
        question_file = os.path.join(data_dir, question_rel_path, "question.jsonl")
        question_edited_file = os.path.join(data_dir, question_rel_path, "question_edited.jsonl")
        
        if not os.path.exists(question_file):
            print(f"Original question file not found: {question_file}, skipping")
            continue
        
        # Load the original question
        questions = load_questions_jsonl(question_file, question_ids=[question_id])
        
        if not questions:
            print(f"Could not find question with ID {question_id} in {question_file}, skipping")
            continue
        
        question = questions[0]
        
        # Read edited files
        prompt_file = os.path.join(edit_dir, "prompt.md")
        ground_truth_file = os.path.join(edit_dir, "ground_truth.py")
        tests_file = os.path.join(edit_dir, "tests.py")
        
        if not os.path.exists(prompt_file) or not os.path.exists(ground_truth_file):
            print(f"Missing required edited files in {edit_dir}, skipping")
            continue
        
        with open(prompt_file, "r") as f:
            prompt = f.read()
        
        with open(ground_truth_file, "r") as f:
            ground_truth = f.read()
        
        # Update the question with edited content
        question["turns"] = [prompt]
        question["ground_truth"] = ground_truth
        
        if os.path.exists(tests_file):
            with open(tests_file, "r") as f:
                tests = f.read()
            question["tests"] = tests
        
        # Load existing edited questions if present
        edited_questions = []
        if os.path.exists(question_edited_file):
            edited_questions = load_questions_jsonl(question_edited_file)
        
        # Check if current question already exists in edited file
        replaced = False
        for i, q in enumerate(edited_questions):
            if q["question_id"] == question_id:
                edited_questions[i] = question
                replaced = True
                break
        
        if not replaced:
            edited_questions.append(question)
        
        # Save updated questions to question_edited.jsonl
        os.makedirs(os.path.dirname(question_edited_file), exist_ok=True)
        with open(question_edited_file, "w") as f:
            for q in edited_questions:
                f.write(json.dumps(q) + "\n")
        
        print(f"Saved edited question {question_id} to {question_edited_file}")
        
        # Delete the edit directory
        import shutil
        shutil.rmtree(edit_dir)
        print(f"Deleted edit directory {edit_dir}")


def main():
    parser = argparse.ArgumentParser(description="Edit questions in LiveBench dataset")
    subparsers = parser.add_subparsers(dest="mode", help="Mode of operation")
    
    # Prepare for edit mode
    prepare_parser = subparsers.add_parser("prepare", help="Prepare questions for editing")
    prepare_parser.add_argument("--question-id", nargs="+", required=True, 
                                help="List of question IDs to prepare for editing")
    
    # Evaluate mode
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate edited questions")
    evaluate_parser.add_argument("--question-id", required=True, 
                                help="Question ID to evaluate")
    evaluate_parser.add_argument("--model", 
                                help="Model to evaluate against instead of ground truth")
    
    # Save edit mode
    save_parser = subparsers.add_parser("save", help="Save edited questions")
    save_parser.add_argument("--question-id", nargs="+", required=True,
                             help="List of question IDs to save edits for")
    
    args = parser.parse_args()
    
    if args.mode == "prepare":
        prepare_edit(args.question_id)
    elif args.mode == "evaluate":
        evaluate(args.question_id, args.model)
    elif args.mode == "save":
        save_edit(args.question_id)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
