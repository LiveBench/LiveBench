#!/usr/bin/env python3
"""
Script to auto-error all questions for which a model doesn't yet have an answer.
Finds all answer files for a given model and generates API error answers for missing questions.
"""

import argparse
import json
import os
import sys
from pathlib import Path
import uuid
import time

from livebench.model.completions import API_ERROR_OUTPUT


def load_jsonl(file_path: Path) -> list[dict]:
    """Load a JSONL file and return list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(file_path: Path, data: list[dict]) -> None:
    """Save a list of dictionaries to a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def generate_error_answer(question_id: int, model_id: str) -> dict:
    """Generate an error answer entry for a missing question."""
    return {
        "question_id": question_id,
        "answer_id": str(uuid.uuid4().hex)[:22],
        "model_id": model_id,
        "choices": [{"index": 0, "turns": [API_ERROR_OUTPUT]}],
        "tstamp": time.time()
    }


def find_model_answer_files(data_dir: Path, model_name: str) -> list[Path]:
    """Find all answer files for a given model in the data directory."""
    answer_files = []
    
    # Search recursively for <model_name>.jsonl files in model_answer directories
    for model_answer_dir in data_dir.rglob("model_answer"):
        answer_file = model_answer_dir / f"{model_name}.jsonl"
        if answer_file.exists():
            answer_files.append(answer_file)
    
    return answer_files


def generate_failure_judgment(question: dict, model_name: str, answer_id: str) -> dict:
    """Generate a failure judgment entry for an error answer."""
    task = question.get("task", "unknown")
    category = question.get("category", "unknown")
    
    result = {
        "question_id": question["question_id"],
        "task": task,
        "model": model_name,
        "score": 0,
        "tstamp": time.time(),
        "category": category,
        "answer_id": answer_id
    }
    
    if "subtask" in question:
        result["subtask"] = question["subtask"]
    
    return result


def process_answer_file(answer_file: Path, model_name: str, dry_run: bool = False) -> tuple[int, int]:
    """
    Process a single answer file and add error answers for missing questions.
    Also creates failure judgments for the error answers.
    Returns tuple of (num_existing, num_added).
    """
    # Find corresponding question.jsonl in parent directory
    question_file = answer_file.parent.parent / "question.jsonl"
    
    if not question_file.exists():
        print(f"Warning: No question.jsonl found for {answer_file}")
        return 0, 0
    
    # Load questions and answers
    questions = load_jsonl(question_file)
    answers = load_jsonl(answer_file)
    
    # Create question lookup by id
    questions_by_id = {q["question_id"]: q for q in questions}
    
    # Get question IDs
    question_ids = {q["question_id"] for q in questions}
    answer_ids = {a["question_id"] for a in answers}
    
    # Find missing question IDs
    missing_ids = sorted(question_ids - answer_ids)
    
    if not missing_ids:
        print(f"✓ {answer_file.relative_to(answer_file.parents[3])}: All {len(question_ids)} questions have answers")
        return len(answer_ids), 0
    
    print(f"• {answer_file.relative_to(answer_file.parents[3])}: {len(answer_ids)}/{len(question_ids)} answered, adding {len(missing_ids)} error answers")
    
    # Generate error answers for missing questions
    new_answers = [generate_error_answer(qid, model_name) for qid in missing_ids]
    
    # Combine and sort by question_id
    all_answers = answers + new_answers
    all_answers.sort(key=lambda x: x["question_id"])
    
    # Save updated answers (unless dry run)
    if not dry_run:
        save_jsonl(answer_file, all_answers)
        print(f"  → Saved {len(all_answers)} total answers to {answer_file.name}")
    else:
        print(f"  → [DRY RUN] Would save {len(all_answers)} total answers")
    
    # Generate and save failure judgments
    judgment_dir = answer_file.parent.parent / "model_judgment"
    judgment_file = judgment_dir / "ground_truth_judgment.jsonl"
    
    if not dry_run:
        # Create judgment directory if it doesn't exist
        judgment_dir.mkdir(exist_ok=True)
        
        # Load existing judgments to avoid duplicates
        existing_judgments = {}
        if judgment_file.exists():
            existing_data = load_jsonl(judgment_file)
            for j in existing_data:
                if "answer_id" in j:
                    existing_judgments[j["answer_id"]] = j
        
        # Add new failure judgments for error answers
        for answer in new_answers:
            judgment = generate_failure_judgment(
                questions_by_id[answer["question_id"]],
                model_name,
                answer["answer_id"]
            )
            existing_judgments[answer["answer_id"]] = judgment
        
        # Sort and save all judgments
        all_judgments = sorted(existing_judgments.values(), key=lambda x: (x["question_id"], x["model"]))
        save_jsonl(judgment_file, all_judgments)
        print(f"  → Saved {len(new_answers)} failure judgments to {judgment_file.name}")
    else:
        print(f"  → [DRY RUN] Would save {len(new_answers)} failure judgments")
    
    return len(answer_ids), len(missing_ids)


def main():
    parser = argparse.ArgumentParser(
        description="Auto-error missing questions for a model"
    )
    parser.add_argument(
        "model_name",
        type=str,
        help="Name of the model (matches the .jsonl filename)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to data directory (defaults to livebench/data)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    
    args = parser.parse_args()
    
    # Determine data directory
    if args.data_dir:
        data_dir = args.data_dir
    else:
        # Default to livebench/data (which is a symlink to livebench-private/data)
        script_dir = Path(__file__).parent
        data_dir = script_dir.parent / "data"
    
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)
    
    print(f"Model: {args.model_name}")
    print(f"Data directory: {data_dir}")
    if args.dry_run:
        print("DRY RUN MODE - No files will be modified\n")
    print()
    
    # Find all answer files for this model
    answer_files = find_model_answer_files(data_dir, args.model_name)
    
    if not answer_files:
        print(f"No answer files found for model: {args.model_name}")
        print(f"Searched in: {data_dir}")
        sys.exit(1)
    
    print(f"Found {len(answer_files)} answer file(s) for {args.model_name}\n")
    
    # Process each answer file
    total_existing = 0
    total_added = 0
    
    for answer_file in sorted(answer_files):
        existing, added = process_answer_file(answer_file, args.model_name, args.dry_run)
        total_existing += existing
        total_added += added
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Summary for {args.model_name}:")
    print(f"  Existing answers: {total_existing}")
    print(f"  Error answers added: {total_added}")
    print(f"  Total answers: {total_existing + total_added}")
    
    if args.dry_run:
        print("\nDRY RUN - No changes were made")
        print("Run without --dry-run to actually update the files")


if __name__ == "__main__":
    main()


