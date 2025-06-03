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

# todo: find a better solution than all these imports.
from livebench.model import get_model_config
from livebench.process_results.data_analysis.tablereformat.utils import table_process_results
from livebench.process_results.data_analysis.cta.utils import cta_process_results
from livebench.process_results.data_analysis.tablejoin.utils import joinmap_process_results
from livebench.process_results.reasoning.web_of_lies_v2.utils import web_of_lies_process_results
from livebench.process_results.reasoning.house_traversal.utils import house_traversal_process_results
from livebench.process_results.reasoning.zebra_puzzle.utils import get_zebra_puzzle_evaluator
from livebench.process_results.reasoning.spatial.utils import spatial_process_results
from livebench.process_results.math.math_competitions.utils import mathcontest_process_results,aime_process_results 
from livebench.process_results.math.olympiad.utils import proof_rearrangement_process_results
from livebench.process_results.math.AMPS_Hard.utils import amps_hard_process_results 
from livebench.process_results.writing.plot_unscrambling.utils import plot_unscrambling_process_results
from livebench.process_results.writing.typos.utils import typos_process_results
from livebench.process_results.writing.connections.utils import get_connections_puzzle_evaluator
from livebench.process_results.coding.utils import LCB_generation_process_results, code_generation_process_results, agentic_coding_process_results
from livebench.process_results.instruction_following.utils import instruction_following_process_results
from livebench.process_results.reasoning.web_of_lies_v3.utils import web_of_lies_v3_process_results
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
    LIVE_BENCH_DATA_SUPER_PATH
)


def reorg_output_file(output_file):
    """De-duplicate and sort by question id and model"""
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

    try:
        if task_or_subtask == 'math_comp':
            splits = task_or_subtask.split('_')
            if splits[0] in ["amc", "smc"] or (len(splits) > 1 and splits[1] == "amc"):
                score = mathcontest_process_results(ground_truth, llm_answer, question_text, debug)
                category = "math"
            elif splits[0] == "aime":
                score = aime_process_results(ground_truth, llm_answer, debug)
                category = "math"
            elif splits[0] in ["imo", "usamo"]:
                score = proof_rearrangement_process_results(ground_truth, llm_answer, edit_distance=True, debug=debug)
                category = "math"
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
        elif "amps_hard" in task_or_subtask:
            score = amps_hard_process_results(ground_truth, llm_answer, debug)
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
                score = agentic_coding_process_results(question, answer, debug)
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
    output_file: str,
    answer_dir: str,
    model_list: list[str] | None,
    remove_existing_file: bool,
    bench_name: str,
    debug=False,
    ignore_missing_answers=False,
    resume=False
):
    """
    Evaluate answers to questions for all the given models, compared to the expected ground truth answer for each question.

    Args:
        questions: The list of questions to which answers will be evaluated
        output_file: The path to the file where judgments will be written
        answer_dir: The directory containing a file for each model's answers (e.g. {answer_dir}/gpt-4o-mini.jsonl contains answers from gpt-4o-mini)
        model_list: The list of model names whose answers will be evaluated
        remove_existing_file: Whether to remove an existing judgment output file or append
        bench_name: The subset of LiveBench for which answers should be evaluated (e.g. 'live_bench' or 'live_bench/coding')
        parallel: The number of concurrent threads to use for evaluating answers
        resume: When true, skip question-model pairs that already have judgments in the output file
    """


    if model_list is None:
        # evaluate answers for all models who have answers in answer_dir
        models = get_model_list(answer_dir)
        models = [m for m in models if m != 'deepseek-chat']
    else:
        models = model_list

    models = [get_model_config(m).display_name for m in models]

    print('models:', models)

    # Load answers
    model_answers = load_model_answers(answer_dir, models)

    play_a_match_func = play_a_match_gt

    if '/' in output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if output_file and os.path.exists(output_file) and remove_existing_file:
        os.remove(output_file)

    # Load existing judgments if in resume mode
    existing_answer_ids = set()
    if resume and os.path.exists(output_file):
        print(f"Resume mode: Reading existing judgments from {output_file}")
        with open(output_file, "r") as fin:
            for line in fin:
                judgment = json.loads(line)
                # Only track answer_ids - that's all we use for resuming
                if "answer_id" in judgment:
                    existing_answer_ids.add(judgment["answer_id"])
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
            
            # Only skip evaluation if answer has an ID and that ID was already processed
            if answer_id is not None and answer_id in existing_answer_ids:
                continue
            
            # Always include answers without answer_id or with new answer_id
            filtered_matches.append(match)
        
        matches = filtered_matches
        print(f"Resume mode: Filtered out {original_match_count - len(matches)} already judged matches")

    if len(matches) == 0:
        print('No question-answer pairs found to be judged')
        reorg_output_file(output_file)
        return

    match_stat = {}
    match_stat["bench_name"] = bench_name
    match_stat["model_list"] = models
    match_stat["total_num_questions"] = len(questions)
    match_stat["total_num_matches"] = len(matches)
    match_stat["output_path"] = output_file

    # Show match stats and prompt enter to continue
    print("Stats:")
    print(json.dumps(match_stat, indent=4))
    #input("Press Enter to confirm...")

    if "instruction_following" in bench_name:
        # instruction following tasks are evaluated differently from all other tasks
        nltk.download('punkt')
        nltk.download('punkt_tab')
        task_name = matches[0].question['task']

        if model_list is None:
            models = get_model_list(answer_dir)
        else:
            models = model_list

        for m in model_answers:
            for q in model_answers[m]:
                model_answers[m][q]['choices'][0]['turns'][0] = re.sub(f"<think>.*?<\/think>", "", model_answers[m][q]['choices'][0]['turns'][0], flags=re.DOTALL).strip()

        for model_id in models:
            scores = instruction_following_process_results(questions, model_answers, task_name, model_id, debug)
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

                if output_file:
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                    with open(output_file, "a") as fout:
                        fout.write(json.dumps(result) + "\n")
    elif "agentic_coding" in bench_name:
        for model_id in models: # TODO: parallelize at the model level too
            model_matches = [m for m in matches if m.model == model_id]
            questions = [m.question for m in model_matches]
            answers = [m.answer for m in model_matches]
            eval_result = agentic_coding_process_results(questions, answers, debug=debug, max_workers=parallel)
            for question_id in sorted(eval_result.keys()):
                model_answer = model_answers[model_id][question_id]
                question = [q for q in questions if q['question_id'] == question_id][0]
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
                
                if output_file:
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                    with open(output_file, "a") as fout:
                        fout.write(json.dumps(result) + "\n")
    else:
        # Play matches
        # parallel doesn't work well with the livecodebench eval
        if parallel == 1 or bench_name == "live_bench/coding/coding_completion" or bench_name == "live_bench/coding/LCB_generation":
            for match in tqdm(matches):
                results = play_a_match_func(match, output_file=output_file, debug=debug)
        else:

            def play_a_match_wrapper(match):
                return play_a_match_func(match, output_file=output_file, debug=debug)

            np.random.seed(0)
            np.random.shuffle(matches)

            with ThreadPoolExecutor(parallel) as executor:
                for match in tqdm(
                    executor.map(play_a_match_wrapper, matches), total=len(matches)
                ):
                    pass

    # De-duplicate and sort judgment file
    reorg_output_file(output_file)
    