#!/usr/bin/env python3
"""
Replay agent trajectories for specific questions.

This script loads existing trajectories from model answers, replays them in the environment,
and generates new answer files with the replay results.

Usage:
python replay_agent_trajectory.py --model <model_name> [<model_name2> ...] [--question-id <id1> <id2> ...]

Options:
- --model: Name(s) of the model(s) whose trajectories to replay (required, can specify multiple)
- --question-id: List of question IDs to replay (optional - if not provided, replays all questions with no submission and exit status not 'Submitted')
- --parallel-requests: Number of parallel workers for replay (default: 1)
- --parallel-grading: Number of parallel workers for grading (default: 1)
"""

import argparse
import glob
import json
import sys
import subprocess
from pathlib import Path

import shortuuid

from livebench.gen_api_answer import setup_model
from livebench.common import LIVE_BENCH_ROOT_PATH
from livebench.model.api_model_config import get_model_config
from livebench.agentic_code_runner.minisweagent.run_inference import run_agentic_coding_inference


def load_model_answers(model_name: str) -> list[dict]:
    """Load model answers from the specified JSONL file."""
    model_answer_path_glob = f"data/live_bench/agentic_coding/**/{model_name}.jsonl"
    file_paths = glob.glob(model_answer_path_glob, recursive=True)
    
    if not file_paths:
        print(f"Error: Could not find answer file for model {model_name}.")
        sys.exit(1)
    
    answers: list[dict] = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    obj: dict = json.loads(line.strip())
                    obj["__answer_path"] = file_path
                    answers.append(obj)
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line in {file_path}. Skipping.")
                    continue
    
    return answers


def load_questions_from_answer_path(answer_path: str) -> dict[str, dict]:
    """Load questions from question.jsonl file corresponding to an answer path."""
    answer_path_obj = Path(answer_path)
    question_file = answer_path_obj.parent.parent / "question.jsonl"
    if not question_file.exists():
        alt = answer_path_obj.parent / "question.jsonl"
        question_file = alt if alt.exists() else question_file
        if not question_file.exists():
            print(f"Warning: Could not find question.jsonl for {answer_path}")
            return {}
    
    questions_by_id = {}
    try:
        with open(question_file, "r") as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    question_id = str(obj.get("question_id", ""))
                    if question_id:
                        questions_by_id[question_id] = obj
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Warning: Error reading {question_file}: {e}")
    
    return questions_by_id


def process_model(model_name: str, question_ids: list[str] | None, parallel_requests: int, parallel_grading: int):
    """Process a single model for replay."""
    print(f"\n{'='*80}")
    print(f"Processing model: {model_name}")
    print(f"{'='*80}\n")
    
    print(f"Loading answers for model: {model_name}")
    model_config = get_model_config(model_name)
    model_display_name = model_config.display_name
    
    all_answers = load_model_answers(model_display_name)
    print(f"Loaded {len(all_answers)} total answers")
    
    # Filter to requested question IDs or find questions with no submission
    if question_ids:
        question_ids_set = set(question_ids)
        filtered_answers = [ans for ans in all_answers if str(ans.get('question_id', '')) in question_ids_set]
        print(f"Filtering to requested question IDs")
    else:
        # Find all questions with no submission and exit status not "Submitted"
        filtered_answers = []
        for ans in all_answers:
            # Check trajectory exit status
            trajectory_raw = ans.get("trajectory")
            if trajectory_raw:
                # Parse trajectory if it's a string
                trajectory = None
                if isinstance(trajectory_raw, str):
                    try:
                        trajectory = json.loads(trajectory_raw)
                    except json.JSONDecodeError:
                        pass
                elif isinstance(trajectory_raw, dict):
                    trajectory = trajectory_raw
                
                # Check if exit status is "Submitted"
                if trajectory and isinstance(trajectory, dict):
                    exit_status = trajectory.get('info', {}).get('exit_status')
                    if exit_status == "Submitted":
                        continue
            
            # Check if there's no submission in choices/turns
            choices = ans.get('choices', [])
            if choices and len(choices) > 0:
                turns = choices[0].get('turns', [])
                if turns and len(turns) > 0:
                    if turns[0] == '':
                        filtered_answers.append(ans)
                else:
                    # No turns means no submission
                    filtered_answers.append(ans)
            else:
                # No choices means no submission
                filtered_answers.append(ans)
        print(f"Found {len(filtered_answers)} questions with no submission and exit status not 'Submitted'")
        question_ids = [ans.get('question_id', '') for ans in filtered_answers]
    
    if not filtered_answers:
        print(f"No answers found to replay for model {model_name}")
        return
    
    print(f"Found {len(filtered_answers)} answers to replay")
    
    # Load questions to get instance_id mapping
    questions_by_path = {}
    for answer in filtered_answers:
        answer_path = answer.get("__answer_path")
        if answer_path and answer_path not in questions_by_path:
            questions_by_path[answer_path] = load_questions_from_answer_path(answer_path)
    
    # Create replay trajectory directory
    replay_run_id = f"{model_display_name}_{shortuuid.uuid()}"
    replay_traj_dir = LIVE_BENCH_ROOT_PATH / "agentic_code_runner/data/replay_trajectories" / replay_run_id
    replay_traj_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving trajectories to: {replay_traj_dir}")
    
    # Extract and save trajectories
    questions_to_replay = []
    instance_id_to_question_id = {}
    
    for answer in filtered_answers:
        question_id = str(answer.get('question_id', ''))
        answer_path = answer.get("__answer_path")
        
        # Get the question object to find instance_id
        question = None
        if answer_path and answer_path in questions_by_path:
            question = questions_by_path[answer_path].get(question_id)
        
        if not question:
            print(f"Warning: Could not find question object for question_id {question_id}, skipping")
            continue
        
        instance_id = question.get('instance_id')
        if not instance_id:
            print(f"Warning: Question {question_id} has no instance_id, skipping")
            continue
        
        instance_id_to_question_id[instance_id] = question_id
        
        # Extract trajectory
        trajectory_raw = answer.get("trajectory")
        if not trajectory_raw:
            print(f"Warning: No trajectory found for question_id {question_id}, skipping")
            continue
        
        # Parse trajectory if it's a string
        if isinstance(trajectory_raw, str):
            try:
                trajectory = json.loads(trajectory_raw)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse trajectory for question_id {question_id}, skipping")
                continue
        elif isinstance(trajectory_raw, dict):
            trajectory = trajectory_raw
        else:
            print(f"Warning: Unexpected trajectory format for question_id {question_id}, skipping")
            continue
        
        # Save trajectory file
        traj_file = replay_traj_dir / f"{question_id}.traj.json"
        with open(traj_file, 'w') as f:
            json.dump(trajectory, f, indent=2)
        
        print(f"Saved trajectory for question_id {question_id}")
        
        # Add question to replay list
        questions_to_replay.append(question)
    
    if not questions_to_replay:
        print("No valid trajectories to replay")
        return
    
    print(f"\nReplaying {len(questions_to_replay)} trajectories...")
    
    # Run inference with replay mode
    # For batch mode or multiple questions, pass the directory
    # For single mode with one question, pass the specific file
    if parallel_requests == 1 and len(questions_to_replay) == 1:
        replay_traj_path = str(replay_traj_dir / f"{questions_to_replay[0]['question_id']}.traj.json")
        effective_parallel = 1
    else:
        # Use batch mode for multiple questions or when parallel > 1
        replay_traj_path = str(replay_traj_dir)
        effective_parallel = max(parallel_requests, 1)
    
    provider, api_kwargs, api_name, api_dict = setup_model(model_config)
    
    run_agentic_coding_inference(
        questions=questions_to_replay,
        model_api_name=api_name,
        provider=provider,
        force_temperature=None,
        num_choices=1,
        model_api_kwargs=api_kwargs,
        api_dict=api_dict,
        model_display_name_override=model_display_name,
        answer_file=None,  # We'll write answer files manually after
        parallel=effective_parallel,
        replay_traj_dir=replay_traj_path,
        custom_run_id=replay_run_id,
    )
    
    print("\nReplay complete. Collecting results and updating answer files...")
    
    # Now collect the new trajectories and update answer files
    output_traj_dir = LIVE_BENCH_ROOT_PATH / "agentic_code_runner/data/trajectories" / replay_run_id
    
    # Group answers by their original answer file
    answers_by_file = {}
    for answer in filtered_answers:
        answer_path = answer.get("__answer_path")
        if answer_path:
            if answer_path not in answers_by_file:
                answers_by_file[answer_path] = []
            answers_by_file[answer_path].append(answer)
    
    # Update each answer file
    for answer_file_path, answers_for_file in answers_by_file.items():
        print(f"\nUpdating answer file: {answer_file_path}")
        
        # Load all existing answers from the file (not just the ones we replayed)
        # We need to preserve all answers, only updating the specific ones we replayed
        existing_answers = []
        with open(answer_file_path, 'r') as f:
            for line in f:
                try:
                    existing_answers.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        
        # Create a map of question_id to index
        answer_index = {str(ans.get('question_id', '')): i for i, ans in enumerate(existing_answers)}
        
        # Update with new replay results
        for answer in answers_for_file:
            question_id = str(answer.get('question_id', ''))
            
            # Get instance_id
            answer_path = answer.get("__answer_path")
            question = None
            if answer_path and answer_path in questions_by_path:
                question = questions_by_path[answer_path].get(question_id)
            
            if not question:
                continue
            
            instance_id = question.get('instance_id')
            if not instance_id:
                continue
            
            # Load new trajectory
            new_traj_file = output_traj_dir / str(question_id) / f"{question_id}.traj.json"
            if not new_traj_file.exists():
                print(f"Warning: New trajectory file not found for {question_id}")
                continue
            
            with open(new_traj_file, 'r') as f:
                new_trajectory = json.load(f)
            
            final_answer = new_trajectory['info'].get('submission', '')
            if final_answer is None:
                final_answer = ""
            
            # Remove submission from trajectory before storing
            trajectory_copy = json.loads(json.dumps(new_trajectory))
            if 'submission' in trajectory_copy['info']:
                del trajectory_copy['info']['submission']
            
            # Update the answer
            if question_id in answer_index:
                idx = answer_index[question_id]
                existing_answers[idx].update({
                    'answer_id': shortuuid.uuid(),
                    'trajectory': json.dumps(trajectory_copy, indent=4),
                    'choices': [{'turns': [final_answer]}],
                    'total_output_tokens': new_trajectory['info']['model_stats']['total_output_tokens'],
                    'total_input_tokens': new_trajectory['info']['model_stats']['total_input_tokens'],
                })
                print(f"Updated answer for question_id {question_id}")
        
        # Write back all answers
        with open(answer_file_path, 'w') as f:
            for ans in existing_answers:
                f.write(json.dumps(ans) + '\n')
    
    print("\nDone! All answer files updated.")
    
    # Run grading
    print(f"\nRunning grading for {len(filtered_answers)} questions...")
    
    grading_cmd = [
        'python',
        'run_livebench.py',
        '--model',
        model_name,
        '--question-source',
        'jsonl',
        '--question-id',
        *question_ids,
        '--skip-inference',
        '--parallel-grading',
        str(parallel_grading),
    ]
    
    print(f"Running command: {' '.join(grading_cmd)}")
    
    try:
        subprocess.run(grading_cmd, check=True)
        print("\nGrading complete!")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Grading subprocess failed with error: {e}")
    except KeyboardInterrupt:
        print("\nGrading interrupted by user.")


def main():
    parser = argparse.ArgumentParser(description="Replay agent trajectories for specific questions.")
    parser.add_argument("--model", required=True, nargs='+', help="Name(s) of the model(s) whose trajectories to replay")
    parser.add_argument("--question-id", nargs='+', help="List of question IDs to replay (optional - if not provided, replays all questions with no submission and exit status not 'Submitted')")
    parser.add_argument("--parallel-requests", type=int, default=1, help="Number of parallel workers for replay (default: 1)")
    parser.add_argument("--parallel-grading", type=int, default=1, help="Number of parallel workers for grading (default: 1)")
    args = parser.parse_args()

    for model_name in args.model:
        try:
            process_model(model_name, args.question_id, args.parallel_requests, args.parallel_grading)
        except Exception as e:
            print(f"\nError processing model {model_name}: {e}")
            import traceback
            traceback.print_exc()
            print(f"Continuing to next model...\n")


if __name__ == "__main__":
    main()

