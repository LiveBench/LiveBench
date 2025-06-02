#!/usr/bin/env python3
"""
Script to test different system prompts on LiveBench tasks.

Usage:
python3 test_prompts.py --bench-name live_bench/coding/code_generation --prompt-file prompts.jsonl --model gpt-3.5-turbo
"""

import argparse
import json
import os
import copy
from typing import List, Dict, Any

from livebench.common import load_questions_jsonl, LIVE_BENCH_RELEASES
from livebench.model import get_model_config
from livebench.gen_api_answer import run_questions
from livebench.gen_ground_truth_judgment import gen_judgments


def load_prompts(prompt_file: str) -> List[str]:
    """Load prompts from a JSONL file."""
    prompts = []
    with open(prompt_file, 'r') as f:
        for line in f:
            if line.strip():
                prompt_data = json.loads(line)
                prompts.append(prompt_data['prompt'])
    return prompts


def create_question_sets(questions: List[Dict[str, Any]], prompts: List[str]) -> List[List[Dict[str, Any]]]:
    """Create question sets for each prompt."""
    question_sets = []
    
    for prompt_index, prompt in enumerate(prompts):
        # Create a deep copy of questions for this prompt
        prompt_questions = copy.deepcopy(questions)
        
        # Update each question with the system prompt and task index
        for question in prompt_questions:
            question['system_prompt'] = prompt
            question['task'] = question['task'] + '_' + str(prompt_index)
        
        question_sets.append(prompt_questions)
    
    return question_sets


def main():
    parser = argparse.ArgumentParser(
        description="Test different system prompts on LiveBench tasks"
    )
    
    # Required arguments
    parser.add_argument(
        "--bench-name",
        type=str,
        required=True,
        help="The name of the benchmark task to evaluate (e.g., live_bench/coding/code_generation)"
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        required=True,
        help="Path to JSONL file containing prompts. Each object should have a 'prompt' attribute."
    )
    
    # Arguments from gen_api_answer.py
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="Model to use for evaluation")
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="If provided, will be used as the base of an openai API request, along with the environment variable LIVEBENCH_API_KEY",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--force-temperature", 
        type=float, 
        help="Forcibly set a sampling temperature."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", 
        type=int, 
        help="A debug option. The end index of questions."
    )
    parser.add_argument(
        "--parallel", 
        type=int, 
        default=1, 
        help="The number of concurrent API calls."
    )
    parser.add_argument(
        "--livebench-release-option",
        type=str,
        default=max(LIVE_BENCH_RELEASES),
        choices=sorted(LIVE_BENCH_RELEASES),
        help="Livebench release to use. Provide a single date option.",
    )
    parser.add_argument(
        "--question-id",
        type=str,
        default=None,
        nargs="+",
        help="A list of question ids to generate answers for.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Do not generate answers for questions that have already been generated."
    )
    parser.add_argument(
        "--model-display-name",
        type=str,
        default=None,
        help="Optional display name of the model. If not provided, will be inferred from --model.",
    )
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        default=False,
        help="Retry generating answers for questions that have failed in the past.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        default=False,
        help="Stream responses, for models that support streaming"
    )
    parser.add_argument(
        "--model-provider-override",
        type=str,
        default=None,
        help="Override the provider for the model. If not provided, will be inferred from --model.",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        default=False
    )
    
    args = parser.parse_args()
    
    # Validate release option
    if args.livebench_release_option not in LIVE_BENCH_RELEASES:
        raise ValueError(f"Bad release {args.livebench_release_option}.")
    
    release_set = set(
        [r for r in LIVE_BENCH_RELEASES if r <= args.livebench_release_option]
    )
    
    # Set up API configuration
    if args.api_base is not None:
        api_key = os.environ.get("LIVEBENCH_API_KEY", "EMPTY")
        api_dict = {
            "api_key": api_key,
            "api_base": args.api_base,
        }
    else:
        api_dict = None
    
    # Get model configuration
    model_config = get_model_config(args.model)
    model_display_name = args.model_display_name if args.model_display_name else model_config.display_name
    
    # Load questions from the specified benchmark
    question_file = f"data/{args.bench_name}/question.jsonl"
    if not os.path.exists(question_file):
        raise FileNotFoundError(f"Question file not found: {question_file}")
    
    print(f"Loading questions from {question_file}")
    questions = load_questions_jsonl(
        question_file, 
        release_set, 
        args.livebench_release_option, 
        args.question_id
    )
    
    # Apply question range filtering if specified
    questions = questions[args.question_begin:args.question_end]
    
    print(f"Loaded {len(questions)} questions")
    
    # Load prompts
    if not os.path.exists(args.prompt_file):
        raise FileNotFoundError(f"Prompt file not found: {args.prompt_file}")
    
    print(f"Loading prompts from {args.prompt_file}")
    prompts = load_prompts(args.prompt_file)
    print(f"Loaded {len(prompts)} prompts")
    
    # Create question sets for each prompt
    question_sets = create_question_sets(questions, prompts)
    
    # Create output directory
    output_dir = f"prompt_testing/{args.bench_name}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/model_judgment", exist_ok=True)
    
    # # Process each prompt
    answer_files = []
    for prompt_index, prompt_questions in enumerate(question_sets):
        print(f"\nProcessing prompt {prompt_index + 1}/{len(prompts)}")
        print(f"Prompt: {prompts[prompt_index][:100]}...")
        
        # Set answer file path
        answer_file = f"{output_dir}/{prompt_index}/{model_display_name.lower()}.jsonl"
        answer_files.append(answer_file)
        
        if not args.skip_inference:
            print(f"Output to {answer_file}")
            
            # Run questions for this prompt
            run_questions(
                parallel=args.parallel,
                questions=prompt_questions,
                model_config=model_config,
                num_choices=args.num_choices,
                max_tokens=args.max_tokens,
                answer_file=answer_file,
                api_dict=api_dict,
                stream=args.stream,
                force_temperature=args.force_temperature,
                model_provider_override=args.model_provider_override,
                model_display_name_override=model_display_name,
                bench_name=args.bench_name
            )
    
    print(f"\nAll prompts processed. Generated {len(answer_files)} answer files.")
    
    # Generate ground truth judgments
    print("\nGenerating ground truth judgments...")
    
    # Get all model names from answer files
    
    # Load all questions for judgment (using original questions without prompt modifications)
    for prompt_index, prompt_questions in enumerate(question_sets):
        output_file = f"{output_dir}/{prompt_index}/model_judgment/ground_truth_judgment.jsonl"
        answer_dir = f"{output_dir}/{prompt_index}"
        gen_judgments(
            questions=prompt_questions,
            output_file=output_file,
            answer_dir=answer_dir,
            bench_name=args.bench_name,
            debug=False,
            remove_existing_file=False,
            ignore_missing_answers=False,
            resume=args.resume,
            model_list=[model_display_name],
            parallel=1
        )
    
    print("Prompt testing complete!")


if __name__ == "__main__":
    main()