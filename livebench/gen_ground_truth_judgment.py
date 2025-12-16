"""
Usage:
python gen_ground_truth_judgment.py --bench-name live_bench --model-list [LIST-OF-MODEL-ID] --parallel [num-concurrent-api-call] --mode [single|pairwise-baseline|pairwise-all]
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import time
import os
import re
import glob

import nltk
import numpy as np
from tqdm import tqdm

# Number of times to retry failing agentic coding questions
AGENTIC_CODING_RETRIES = 1

# todo: find a better solution than all these imports.
from livebench.model import get_model_config
from livebench.process_results.data_analysis.tablereformat.utils import table_process_results
from livebench.process_results.data_analysis.cta.utils import cta_process_results
from livebench.process_results.data_analysis.tablejoin.utils import joinmap_process_results
from livebench.process_results.reasoning.web_of_lies_v2.utils import web_of_lies_process_results
from livebench.process_results.reasoning.house_traversal.utils import house_traversal_process_results
from livebench.process_results.reasoning.zebra_puzzle.utils import get_zebra_puzzle_evaluator
from livebench.process_results.reasoning.spatial.utils import spatial_process_results
from livebench.process_results.reasoning.sudoku.utils import sudoku_process_results
from livebench.process_results.math.math_competitions.utils import mathcontest_process_results,aime_process_results 
from livebench.process_results.math.olympiad.utils import proof_rearrangement_process_results
from livebench.process_results.math.AMPS_Hard.utils import amps_hard_process_results 
from livebench.process_results.writing.plot_unscrambling.utils import plot_unscrambling_process_results
from livebench.process_results.writing.typos.utils import typos_process_results
from livebench.process_results.writing.connections.utils import get_connections_puzzle_evaluator
from livebench.process_results.coding.utils import LCB_generation_process_results, code_generation_process_results, agentic_coding_process_results
from livebench.process_results.instruction_following.utils import instruction_following_process_results, ifbench_process_results
from livebench.process_results.reasoning.web_of_lies_v3.utils import web_of_lies_v3_process_results
from livebench.process_results.reasoning.theory_of_mind.utils import theory_of_mind_process_results
from livebench.process_results.reasoning.logic_with_navigation.utils import logic_with_navigation_process_results
from livebench.common import (
    LIVE_BENCH_RELEASES,
    load_questions,
    load_questions_jsonl,
    load_model_answers,
    check_data,
    get_model_list,
    load_test_cases_jsonl,
    make_match_single,
    MatchSingle,
    get_categories_tasks,
    LIVE_BENCH_DATA_SUPER_PATH,
    check_agentic_coding_requirements
)


def reorg_output_file(output_file):
    """De-duplicate and sort by question id and model"""
    if not os.path.exists(output_file):
        return
    
    judgments = {}
    with open(output_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            model = json.loads(l)["model"]
            key = (qid, model)
            judgments[key] = l

    keys = sorted(list(judgments.keys()))
    with open(output_file, "w") as fout:
        for key in keys:
            fout.write(judgments[key])


def play_a_match_gt(match: MatchSingle, output_file: str | None = None, debug=False):
    """
    Evaluate a model's answer to a question.

    Args:
        match: An object containing the question, model name, and model answer.
        output_file: The path to which the judgment should be outputted.
    
    Returns:
        result: The judgment, containing the question id, task name, model name, score, turn, timestamp, and category name
    """
    question, model, answer = (
        match.question,
        match.model,
        match.answer,
    )
    coding_test_case_tasks = ["coding_completion", "LCB_generation", "code_generation", "code_completion", "agentic_coding"]
    if "ground_truth" not in question and question["task"] not in coding_test_case_tasks and question["category"] != "instruction_following":
        # aside from coding and instruction following tasks, all questions should contain the ground truth answer
        raise ValueError("Questions must have ground_truth to run gen_ground_truth_judgment.")

    task = question["task"]
    task_or_subtask = question["subtask"] if "subtask" in question.keys() else question["task"]
    question_text = question["turns"][0]
    ground_truth = question.get("ground_truth", None)
    llm_answer = answer['choices'][0]['turns'][-1]
    llm_answer = re.sub(f"<think>.*?<\/think>", "", llm_answer, flags=re.DOTALL)
    score = 0
    category = None

    # todo: find a better solution than a long if statement.

    splits = task_or_subtask.split('_')

    try:
        if question.get("category") == "instruction_following":
            # IFBench format questions (old format never reaches play_a_match_gt)
            score = ifbench_process_results(question, llm_answer, debug)
            category = "instruction_following"
        elif len(splits) > 0 and (splits[0] in ["amc", "smc", "aime", "imo", "usamo"] or (len(splits) > 1 and splits[1] == "amc")):
            if splits[0] in ["amc", "smc"] or (len(splits) > 1 and splits[1] == "amc"):
                score = mathcontest_process_results(ground_truth, llm_answer, question_text, debug)
                category = "math"
            elif splits[0] == "aime":
                score = aime_process_results(ground_truth, llm_answer, debug)
                category = "math"
            elif splits[0] in ["imo", "usamo"]:
                score = proof_rearrangement_process_results(ground_truth, llm_answer, edit_distance=True, debug=debug)
                category = "math"
            else:
                raise Exception("Invalid task or subtask provided: ", question['task'], question['subtask'])
        elif task_or_subtask == "cta":
            score = cta_process_results(ground_truth, llm_answer, debug)
            category = "data_analysis"
        elif task_or_subtask == "tablereformat":
            if question["livebench_release_date"] >= "2025-04-25":
                version = "v2"
            else:
                version = "v1"
            score = table_process_results(question_text, ground_truth, llm_answer, version, debug)
            category = "data_analysis"
        elif task_or_subtask == "tablejoin":
            score = joinmap_process_results(question_text, ground_truth, llm_answer, debug)
            category = "data_analysis"
        elif "amps_hard" in task_or_subtask or "amps_hard" in task:
            score = amps_hard_process_results(ground_truth, llm_answer, debug, question=question)
            category = "math"
        elif task_or_subtask == "web_of_lies_v2" or task_or_subtask == "web_of_lies_v3":
            if task_or_subtask == "web_of_lies_v2":
                score = web_of_lies_process_results(ground_truth, llm_answer, debug)
            else:
                score = web_of_lies_v3_process_results(ground_truth, llm_answer, debug)
            category = "reasoning"
        elif task_or_subtask == "house_traversal":
            score = house_traversal_process_results(ground_truth, llm_answer, debug)
            category = "reasoning"
        elif 'zebra_puzzle' in task_or_subtask:
            zebra_evaluator = get_zebra_puzzle_evaluator(question["livebench_release_date"])
            score = zebra_evaluator(ground_truth, llm_answer, debug)
            category = "reasoning"
        elif task_or_subtask == "spatial":
            score = spatial_process_results(ground_truth, llm_answer, debug)
            category = "reasoning"
        elif task_or_subtask == "theory_of_mind":
            score = theory_of_mind_process_results(ground_truth, llm_answer, debug)
            category = "reasoning"
        elif task_or_subtask == "logic_with_navigation":
            score = logic_with_navigation_process_results(ground_truth, llm_answer, debug)
            category = "reasoning"
        elif task_or_subtask == "sudoku":
            score = sudoku_process_results(ground_truth, llm_answer, debug)
            category = "reasoning"
        elif task_or_subtask == 'typos':
            score = typos_process_results(ground_truth, llm_answer, debug)
            category = "language"
        elif task_or_subtask == "connections":
            connections_evaluator = get_connections_puzzle_evaluator(question["livebench_release_date"])
            score = connections_evaluator(ground_truth, llm_answer, debug)
            category = "language"
        elif task_or_subtask == "plot_unscrambling":
            score = plot_unscrambling_process_results(ground_truth, llm_answer, debug)
            category = "language"
        elif task_or_subtask in coding_test_case_tasks:
            # use entire question object, because there are test cases inside.
            if task_or_subtask == "LCB_generation" or task_or_subtask == "coding_completion":
                score = LCB_generation_process_results(question, llm_answer, debug)
            elif task_or_subtask == "code_generation" or task_or_subtask == "code_completion":
                score = code_generation_process_results(question, llm_answer, debug)
            elif task_or_subtask == "agentic_coding":
                # Check for litellm and Docker availability
                if not check_agentic_coding_requirements():
                    score = 0  # Return 0 score when requirements are not met
                else:
                    # Note: agentic_coding typically uses batch processing in gen_judgments
                    # This single-question path wraps the question/answer in lists
                    eval_result = agentic_coding_process_results([question], [answer], debug)
                    # If the question was skipped due to infrastructure error, return None
                    # to signal that judgment should not be written
                    if question['question_id'] not in eval_result:
                        print(f"Skipping judgment for {question['question_id']} due to infrastructure error")
                        return None
                    score = eval_result[question['question_id']]
                    
                    # Retry if failed
                    for retry_num in range(AGENTIC_CODING_RETRIES):
                        if score == 1:
                            break
                        print(f"Retrying question {question['question_id']} (attempt {retry_num + 1}/{AGENTIC_CODING_RETRIES})")
                        retry_result = agentic_coding_process_results([question], [answer], debug)
                        if question['question_id'] in retry_result and retry_result[question['question_id']] == 1:
                            score = 1
            category = "coding"
        else:
            raise NotImplementedError(f"This task ({task_or_subtask}) has not been implemented yet.")
    except Exception as e:
        raise RuntimeError(f"Error occurred evaluating question {question['question_id']}") from e

    if not category:
        raise NotImplementedError(f"A category must be assigned to each task")
    question_id = question["question_id"]
    result = {
        "question_id": question_id,
        "task": task,
        "model": model,
        "score": score,
        "tstamp": time.time(),
        "category": category,
    }
    # Add answer_id if available
    if "answer_id" in answer:
        result["answer_id"] = answer["answer_id"]
    
    if "subtask" in question.keys():
        result["subtask"] = question["subtask"]
    print(
        f"question: {question_id}, model: {model}, "
        f"score: {score}, "
       
    )

    if output_file:
        if '/' in output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "a") as fout:
            fout.write(json.dumps(result) + "\n")

    return result


def gen_judgments(
    parallel: int,
    questions: list[dict],
    output_file: str | None,
    model_answers: dict[str, dict],
    model_list: list[str] | None,
    remove_existing_file: bool,
    bench_name: str,
    debug=False,
    ignore_missing_answers=False,
    resume=False,
    only_incorrect=False
):
    """
    Evaluate answers to questions for all the given models, compared to the expected ground truth answer for each question.

    Args:
        questions: The list of questions to which answers will be evaluated
        output_file: The path to the file where judgments will be written (can be None if questions have _output_file metadata)
        model_answers: Pre-loaded model answers dict (model_name -> question_id -> answer)
        model_list: The list of model names whose answers will be evaluated
        remove_existing_file: Whether to remove an existing judgment output file or append
        bench_name: The subset of LiveBench for which answers should be evaluated (e.g. 'live_bench' or 'live_bench/coding')
        parallel: The number of concurrent threads to use for evaluating answers
        resume: When true, skip question-model pairs that already have judgments in the output file
        only_incorrect: When true (and resume is true), only re-evaluate questions that previously scored 0
    """

    models = list(model_answers.keys())

    print('models:', models)

    play_a_match_func = play_a_match_gt

    # Collect unique output files from questions
    output_files = set()
    if output_file:
        output_files.add(output_file)
    for question in questions:
        if '_output_file' in question:
            output_files.add(question['_output_file'])
    
    # Create directories and optionally remove existing files
    for out_file in output_files:
        if '/' in out_file:
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
        if out_file and os.path.exists(out_file) and remove_existing_file:
            os.remove(out_file)

    # Load existing judgments if in resume mode
    existing_answer_ids = set()
    existing_scores = {}  # Store answer_id -> score mapping for only_incorrect mode
    if resume:
        for out_file in output_files:
            if os.path.exists(out_file):
                print(f"Resume mode: Reading existing judgments from {out_file}")
                with open(out_file, "r") as fin:
                    for line in fin:
                        judgment = json.loads(line)
                        # Only track answer_ids - that's all we use for resuming
                        if "answer_id" in judgment:
                            answer_id = judgment["answer_id"]
                            existing_answer_ids.add(answer_id)
                            # Store score for only_incorrect mode
                            if only_incorrect and "score" in judgment:
                                existing_scores[answer_id] = judgment["score"]
        if existing_answer_ids:
            print(f"Found {len(existing_answer_ids)} existing answer IDs")

    make_match_func = make_match_single
    if not ignore_missing_answers:
        check_data(questions, model_answers, models)

    # Make matches
    matches = []
    matches += make_match_func(
        questions,
        models,
        model_answers,
        ignore_missing_answers=ignore_missing_answers
    )

    if len(matches) == 0:
        print('No question-answer pairs found')
        return

    # Filter out matches that already have judgments if in resume mode
    if resume:
        original_match_count = len(matches)
        filtered_matches = []

        for match in matches:
            # Check if this answer has already been evaluated
            answer = match.answer
            answer_id = answer.get("answer_id", None)

            # If no answer_id, always include (can't track)
            if answer_id is None:
                filtered_matches.append(match)
                continue

            # If answer_id not in existing judgments, always include
            if answer_id not in existing_answer_ids:
                filtered_matches.append(match)
                continue

            # If we get here, the answer has been evaluated before
            if only_incorrect:
                # Only re-evaluate if the previous score was 0
                previous_score = existing_scores.get(answer_id)
                if previous_score == 0:
                    filtered_matches.append(match)
                # Skip if score was not 0 or if score is missing
            # If not only_incorrect mode, skip all previously evaluated answers

        matches = filtered_matches
        print(f"Resume mode: Filtered out {original_match_count - len(matches)} already judged matches")

    if len(matches) == 0:
        print('No question-answer pairs found to be judged')
        for out_file in output_files:
            reorg_output_file(out_file)
        return

    # Separate matches by category
    agentic_coding_matches = [m for m in matches if m.question.get('category') == 'agentic_coding']
    old_instruction_following_matches = [m for m in matches if m.question.get('category') == 'instruction_following' and m.question.get("livebench_release_date", "") < "2025-11-25"]
    normal_matches = [m for m in matches if m not in agentic_coding_matches and m not in old_instruction_following_matches]

    match_stat = {}
    match_stat["total_num_questions"] = len(questions)
    match_stat["total_num_matches"] = len(matches)
    match_stat["agentic_coding_matches"] = len(agentic_coding_matches)
    match_stat["old_instruction_following_matches"] = len(old_instruction_following_matches)
    match_stat["normal_matches"] = len(normal_matches)
    match_stat["model_list"] = models
    match_stat["output_paths"] = list(output_files)

    # Show match stats
    print("Stats:")
    print(json.dumps(match_stat, indent=4))
    #input("Press Enter to confirm...")

    # Process agentic coding matches if any
    if agentic_coding_matches:
        if not check_agentic_coding_requirements():
            print("Warning: litellm or docker missing, skipping agentic coding evaluation")
        else:
            print(f'Processing {len(agentic_coding_matches)} agentic coding matches')
            for model_id in models:
                model_matches = [m for m in agentic_coding_matches if m.model == model_id]
                if not model_matches:
                    continue
                
                # Initial run
                model_questions = [m.question for m in model_matches]
                answers = [m.answer for m in model_matches]
                eval_result = agentic_coding_process_results(model_questions, answers, debug=debug, max_workers=parallel)
                
                # Retry failing questions
                for retry_num in range(AGENTIC_CODING_RETRIES):
                    # Filter to questions that failed (score == 0)
                    failing_questions = [q for q in model_questions if q['question_id'] in eval_result and eval_result[q['question_id']] == 0]
                    failing_answers = [a for a in answers if a['question_id'] in [q['question_id'] for q in failing_questions]]
                    
                    if not failing_questions:
                        break
                    
                    print(f'Retrying {len(failing_questions)} failing questions (attempt {retry_num + 1}/{AGENTIC_CODING_RETRIES})')
                    retry_result = agentic_coding_process_results(failing_questions, failing_answers, debug=debug, max_workers=parallel)
                    
                    # Update results with any successes
                    for question_id, score in retry_result.items():
                        if score == 1:
                            eval_result[question_id] = 1
                
                # Write final results
                for question_id in sorted(eval_result.keys()):
                    model_answer = model_answers[model_id][question_id]
                    question = [q for q in model_questions if q['question_id'] == question_id][0]
                    result = {
                        "question_id": question_id,
                        "task": question['task'],
                        "model": model_id,
                        "score": eval_result[question_id],
                        "tstamp": time.time(),
                        "category": "agentic_coding",
                    }
                    if "answer_id" in model_answer:
                        result["answer_id"] = model_answer["answer_id"]

                    print(
                        f"question: {question_id}, model: {model_id}, "
                        f"score: {eval_result[question_id]}, ")
                    
                    # Find the output file for this question
                    if '_output_file' in question:
                        out_file = question['_output_file']
                    elif output_file:
                        out_file = output_file
                    else:
                        continue
                        
                    os.makedirs(os.path.dirname(out_file), exist_ok=True)
                    with open(out_file, "a") as fout:
                        fout.write(json.dumps(result) + "\n")
    
    # Process old instruction following matches if any
    if old_instruction_following_matches:
        print(f'Processing {len(old_instruction_following_matches)} old instruction following matches')
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('averaged_perceptron_tagger')
        
        # Get questions for this category
        if_questions = list(set([m.question for m in old_instruction_following_matches]))
        task_name = if_questions[0]['task']

        for m in model_answers:
            for q in model_answers[m]:
                if q in [ques['question_id'] for ques in if_questions]:
                    model_answers[m][q]['choices'][0]['turns'][0] = re.sub(f"<think>.*?<\/think>", "", model_answers[m][q]['choices'][0]['turns'][0], flags=re.DOTALL).strip()

        for model_id in models:
            scores = instruction_following_process_results(if_questions, model_answers, task_name, model_id, debug)
            for item in scores:
                question_id = item["question_id"]
                score = item["score"]
                turn = 1
                result = {
                    "question_id": question_id,
                    "task": task_name,
                    "model": model_id,
                    "score": score,
                    "turn": turn,
                    "tstamp": time.time(),
                    "category": "instruction_following",
                }
                # Add answer_id if available
                answer = model_answers.get(model_id, {}).get(question_id, {})
                if answer and "answer_id" in answer:
                    result["answer_id"] = answer["answer_id"]
                    
                print(
                    f"question: {question_id}, turn: {turn}, model: {model_id}, "
                    f"score: {score}, ")

                # Find the output file for this question
                question_obj = [q for q in if_questions if q['question_id'] == question_id]
                if question_obj and '_output_file' in question_obj[0]:
                    out_file = question_obj[0]['_output_file']
                elif output_file:
                    out_file = output_file
                else:
                    continue
                    
                os.makedirs(os.path.dirname(out_file), exist_ok=True)
                with open(out_file, "a") as fout:
                    fout.write(json.dumps(result) + "\n")
    
    # Process normal matches if any
    if normal_matches:
        print(f'Processing {len(normal_matches)} normal matches')
        # Check if any match has task that requires sequential processing
        has_lcb = any(m.question.get('task') in ['coding_completion', 'LCB_generation'] for m in normal_matches)
        
        # Play matches
        if parallel == 1 or has_lcb:
            for match in tqdm(normal_matches):
                # Use question's output file if available
                question_output_file = match.question.get('_output_file', output_file)
                results = play_a_match_func(match, output_file=question_output_file, debug=debug)
        else:
            def play_a_match_wrapper(match):
                # Use question's output file if available
                question_output_file = match.question.get('_output_file', output_file)
                return play_a_match_func(match, output_file=question_output_file, debug=debug)

            np.random.seed(0)
            np.random.shuffle(normal_matches)

            with ThreadPoolExecutor(parallel) as executor:
                for match in tqdm(
                    executor.map(play_a_match_wrapper, normal_matches), total=len(normal_matches)
                ):
                    pass
    
    # De-duplicate and sort all judgment files
    print(f"Reorganizing {len(output_files)} output files")
    for out_file in output_files:
        reorg_output_file(out_file)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model answers compared to ground truths")
    parser.add_argument(
        "--bench-name",
        type=str,
        nargs="+",
        default="live_bench",
        help="The name(s) of the benchmark question set. Defaults to 'live_bench', or all tasks in the benchmark. Specify e.g. live_bench/reasoning/web_of_lies_v2 to generate only for that task.",
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=None,
        help="A list of models to be evaluated",
    )
    parser.add_argument(
        "--parallel", type=int, default=1, help="The number of concurrent API calls."
    )
    parser.add_argument(
        "--first-n", type=int, help="A debug option. Only run the first `n` judgments."
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument(
        "--remove-existing-file", action="store_true", default=False,
        help="Remove existing judgment file."
    )
    parser.add_argument(
        "--question-source", type=str, default="huggingface", help="The source of the questions. 'huggingface' will draw questions from huggingface. 'jsonl' will gather local jsonl files at data/{bench_name}/**/question.jsonl to permit tweaking or writing custom questions."
    )
    parser.add_argument(
        "--livebench-release-option", 
        type=str, 
        default=max(LIVE_BENCH_RELEASES),
        choices=sorted(LIVE_BENCH_RELEASES),
        help="Livebench release to use. Provide a single date option. Will handle excluding deprecated questions for selected release."
    )
    parser.add_argument(
        "--debug", action="store_true", default=False,
        help="Print debug information."
    )
    parser.add_argument(
        "--question-id", type=str, nargs="+", default=None,
        help="A debug option. The question id to be evaluated."
    )
    parser.add_argument(
        "--exclude-question-id", type=str, nargs="+", default=None,
        help="A debug option. The question id to be excluded from the evaluation."
    )
    parser.add_argument(
        "--model-display-name", type=str, nargs="+",default=None,
        help="The display name of the model(s). If provided, will be used to name the output file. Will match order to --model-list. If not provided, will be generated from --model-list."
    )
    parser.add_argument(
        "--ignore-missing-answers", action="store_true", default=False, help="Don't raise an error if a model is missing answers to some questions"
    )
    parser.add_argument(
        "--answer-file", type=str, default=None
    )
    parser.add_argument(
        "--output-file", type=str, default=None
    )
    parser.add_argument(
        "--resume", action="store_true", default=False,
        help="Resume evaluation, skipping question-model pairs already in the output file."
    )
    parser.add_argument(
        "--only-incorrect", action="store_true", default=False,
        help="When used with --resume, only re-evaluate questions that previously scored 0. Requires --resume to be enabled."
    )
    args = parser.parse_args()

    if args.livebench_release_option not in LIVE_BENCH_RELEASES:
        raise ValueError(f"Bad release {args.livebench_release_option}.")

    # Validate that --only-inorrect requires --resume
    if args.only_incorrect and not args.resume:
        raise ValueError("--only-incorrect can only be used with --resume")

    release_set = set([
        r for r in LIVE_BENCH_RELEASES if r <= args.livebench_release_option
    ])

    if args.model is None:
        model_list = None
    else:
        # model_list = [get_model(model_name).display_name for model_name in args.model_list]
        model_list = []
        for i, model_name in enumerate(args.model):
            if args.model_display_name is not None:
                model_list.append(args.model_display_name[i].lower())
            else:
                model_list.append(get_model_config(model_name).display_name.lower())

    if args.question_source == "huggingface":
        # Collect all questions upfront across all benchmark names
        all_questions = []
        
        for bench_name in args.bench_name:
            categories, tasks = get_categories_tasks(bench_name)

            for category_name, task_names in tasks.items():
                for task_name in task_names:
                    questions = load_questions(categories[category_name], release_set, args.livebench_release_option, task_name, args.question_id)
                    if args.first_n:
                        questions = questions[: args.first_n]
                    questions = questions[args.question_begin:args.question_end]
                    if args.exclude_question_id:
                        questions = [q for q in questions if q['question_id'] not in args.exclude_question_id]

                    task_full_name = f"{LIVE_BENCH_DATA_SUPER_PATH}/{category_name}/{task_name}"
                    output_file = f"data/{task_full_name}/model_judgment/ground_truth_judgment.jsonl" if args.output_file is None else args.output_file
                    answer_dir = f"data/{task_full_name}/model_answer/" if args.answer_file is None else args.answer_file
                    
                    # Attach metadata to each question
                    for question in questions:
                        question['_output_file'] = output_file
                        question['_answer_dir'] = answer_dir
                        question['_task_full_name'] = task_full_name
                    
                    all_questions.extend(questions)
                    print(f"Questions from {task_full_name}")

    elif args.question_source == "jsonl":
        # Collect all questions upfront across all benchmark names
        all_questions = []

        for bench_name in args.bench_name:
            list_of_question_files = []
            original_question_file = f"data/{bench_name}/question.jsonl"
            if os.path.exists(original_question_file):
                list_of_question_files = [original_question_file]
            else:
                list_of_question_files = glob.glob(f"data/{bench_name}/**/question.jsonl", recursive=True)
            
            for question_file in list_of_question_files:
                print('questions from', question_file)
                questions = load_questions_jsonl(question_file, release_set, args.livebench_release_option, args.question_id)
                questions = load_test_cases_jsonl(question_file, questions)
                if args.first_n:
                    questions = questions[: args.first_n]
                
                questions = questions[args.question_begin:args.question_end]
                if args.exclude_question_id:
                    questions = [q for q in questions if q['question_id'] not in args.exclude_question_id]
                bench_name = os.path.dirname(question_file).replace("data/","")

                output_file = f"data/{bench_name}/model_judgment/ground_truth_judgment.jsonl" if args.output_file is None else args.output_file
                answer_dir = f"data/{bench_name}/model_answer/" if args.answer_file is None else args.answer_file
                
                # Attach metadata to each question
                for question in questions:
                    question['_output_file'] = output_file
                    question['_answer_dir'] = answer_dir
                    question['_task_full_name'] = bench_name
                
                all_questions.extend(questions)

    else:
        raise ValueError(f"Bad question source {args.question_source}.")
    
    # Process all questions together
    if not all_questions:
        print("No questions to evaluate")
    else:
        print(f"Running {len(all_questions)} questions together across {len(args.bench_name)} benchmark(s)")
        
        # Gather unique answer directories from all questions
        answer_dirs = set()
        for question in all_questions:
            answer_dirs.add(question['_answer_dir'])
        
        # Load all model answers from all directories
        print(f"Loading model answers from {len(answer_dirs)} directories")
        all_model_answers = {}
        for answer_dir in answer_dirs:
            if model_list is None:
                models = get_model_list(answer_dir)
                models = [m for m in models if m != 'deepseek-chat']
            else:
                models = model_list
            
            models = [get_model_config(m).display_name for m in models]
            
            dir_answers = load_model_answers(answer_dir, models)
            
            # Merge answers - if a model appears in multiple directories, combine them
            for model_name, answers in dir_answers.items():
                if model_name not in all_model_answers:
                    all_model_answers[model_name] = {}
                all_model_answers[model_name].update(answers)
        
        # Process all questions together
        gen_judgments(
            parallel=args.parallel,
            questions=all_questions,
            output_file=None,  # Will use _output_file from question metadata
            model_answers=all_model_answers,
            model_list=model_list,
            remove_existing_file=args.remove_existing_file,
            bench_name='live_bench',
            debug=args.debug,
            ignore_missing_answers=args.ignore_missing_answers,
            resume=args.resume,
            only_incorrect=args.only_incorrect
        )
    