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
from livebench.process_results.data_analysis.tablereformat.utils import table_process_results
from livebench.process_results.data_analysis.cta.utils import cta_process_results
from livebench.process_results.data_analysis.tablejoin.utils import joinmap_process_results
from livebench.process_results.reasoning.web_of_lies_v2.utils import web_of_lies_process_results
from livebench.process_results.reasoning.house_traversal.utils import house_traversal_process_results
from livebench.process_results.reasoning.zebra_puzzle.utils import zebra_puzzle_process_results
from livebench.process_results.reasoning.spatial.utils import spatial_process_results
from livebench.process_results.math.math_competitions.utils import mathcontest_process_results,aime_process_results 
from livebench.process_results.math.olympiad.utils import proof_rearrangement_process_results
from livebench.process_results.math.AMPS_Hard.utils import amps_hard_process_results 
from livebench.process_results.writing.plot_unscrambling.utils import plot_unscrambling_process_results
from livebench.process_results.writing.typos.utils import typos_process_results
from livebench.process_results.writing.connections.utils import connections_process_results
from livebench.process_results.coding.utils import LCB_generation_process_results
from livebench.process_results.instruction_following.utils import instruction_following_process_results

from livebench.common import (
    load_questions,
    load_questions_jsonl,
    load_model_answers,
    check_data,
    get_model_list,
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


def play_a_match_gt(match: MatchSingle, output_file: str):
    question, model, answer = (
        match.question,
        match.model,
        match.answer,
    )
    coding_test_case_tasks = ["coding_completion","LCB_generation"]
    if "ground_truth" not in question and "reference" not in question and question["task"] not in coding_test_case_tasks and question["category"] != "instruction_following":
        raise ValueError("Questions must have ground_truth to run gen_ground_truth_judgment.")

    task = question["task"]
    task_or_subtask = question["subtask"] if "subtask" in question.keys() else question["task"]
    question_text = question["turns"][0]
    ground_truth = question.get("ground_truth", None)
    llm_answer = answer['choices'][0]['turns'][-1]
    score = 0
    category = None

    # todo: find a better solution than a long if statement.
    splits = task_or_subtask.split('_')
    if splits[0] in ["amc", "smc"] or (len(splits) > 1 and splits[1] == "amc"):
        score = mathcontest_process_results(ground_truth, llm_answer, question_text)
        category = "math"
    elif splits[0] == "aime":
        score = aime_process_results(ground_truth, llm_answer)
        category = "math"
    elif splits[0] in ["imo", "usamo"]:
        score = proof_rearrangement_process_results(ground_truth, llm_answer, edit_distance=True)
        category = "math"
    elif task_or_subtask == "cta":
        score = cta_process_results(ground_truth, llm_answer)
        category = "data_analysis"
    elif task_or_subtask == "tablereformat":
        score = table_process_results(question_text, ground_truth, llm_answer)
        category = "data_analysis"
    elif task_or_subtask == "tablejoin":
        score = joinmap_process_results(question_text, ground_truth, llm_answer)
        category = "data_analysis"
    elif "amps_hard" in task_or_subtask:
        score = amps_hard_process_results(ground_truth, llm_answer)
        category = "math"
    elif task_or_subtask == "web_of_lies_v2":
        score = web_of_lies_process_results(ground_truth, llm_answer)
        category = "reasoning"
    elif task_or_subtask == "house_traversal":
        score = house_traversal_process_results(ground_truth, llm_answer)
        category = "reasoning"
    elif task_or_subtask == "zebra_puzzle":
        score = zebra_puzzle_process_results(ground_truth, llm_answer)
        category = "reasoning"
    elif task_or_subtask == "spatial":
        score = spatial_process_results(ground_truth, llm_answer)
        category = "reasoning"
    elif task_or_subtask == 'typos':
        score = typos_process_results(ground_truth, llm_answer)
        category = "language"
    elif task_or_subtask == "connections":
        score = connections_process_results(ground_truth, llm_answer)
        category = "language"
    elif task_or_subtask == "plot_unscrambling":
        score = plot_unscrambling_process_results(ground_truth, llm_answer)
        category = "language"
    elif task_or_subtask in coding_test_case_tasks:
        # use entire question object, because there are test cases inside.
        score = LCB_generation_process_results(question, llm_answer)
        category = "coding"
    else:
        raise NotImplementedError(f"This task ({task_or_subtask}) has not been implemented yet.")

    if not category:
        raise NotImplementedError(f"A category must be assigned to each task")
    question_id = question["question_id"]
    turn = 1
    result = {
        "question_id": question_id,
        "task": task,
        "model": model,
        "score": score,
        "turn": turn,
        "tstamp": time.time(),
        "category": category,
    }
    if "subtask" in question.keys():
        result["subtask"] = question["subtask"]
    print(
        f"question: {question_id}, turn: {turn}, model: {model}, "
        f"score: {score}, "
       
    )

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "a") as fout:
            fout.write(json.dumps(result) + "\n")

    return result


def gen_judgments(
    parallel,
    questions,
    output_file,
    answer_dir,
    model_list,
    remove_existing_file,
    bench_name,
):
    
    # Load answers
    model_answers = load_model_answers(answer_dir)
    print('models:',model_answers.keys())

    if model_list is None:
        models = get_model_list(answer_dir)
    else:
        models = model_list

    play_a_match_func = play_a_match_gt

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if output_file and os.path.exists(output_file) and remove_existing_file:
        os.remove(output_file)

    make_match_func = make_match_single
    check_data(questions, model_answers, models)

    # Make matches
    matches = []
    matches += make_match_func(
        questions,
        models,
        model_answers,
    )

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
        nltk.download('punkt')
        task_name = matches[0].question['task']

        if model_list is None:
            models = get_model_list(answer_dir)
        else:
            models = model_list

        for model_id in models:
            scores = instruction_following_process_results(questions, model_answers, task_name, model_id)
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
                print(
                    f"question: {question_id}, turn: {turn}, model: {model_id}, "
                    f"score: {score}, ")

                if output_file:
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                    with open(output_file, "a") as fout:
                        fout.write(json.dumps(result) + "\n")
    else:
        # Play matches
        if parallel == 1:
            for match in tqdm(matches):
                results = play_a_match_func(match, output_file=output_file)
        else:

            def play_a_match_wrapper(match):
                play_a_match_func(match, output_file=output_file)

            np.random.seed(0)
            np.random.shuffle(matches)

            with ThreadPoolExecutor(parallel) as executor:
                for match in tqdm(
                    executor.map(play_a_match_wrapper, matches), total=len(matches)
                ):
                    pass

    # De-duplicate and sort judgment file
    reorg_output_file(output_file)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench-name",
        type=str,
        default="live_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--model-list",
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
        "--question-source", type=str, default="huggingface", help="The source of the questions. 'huggingface' will draw questions from huggingface. 'jsonl' will use local jsonl files to permit tweaking or writing custom questions."
    )
    parser.add_argument(
        "--livebench-release-option", type=str, default='2024-08-31', help="Livebench release to use. Provide a single date option, current options are {'2024-08-31' (august update), '2024-07-26' (july update), '2024-06-24' (original release)}. Will handle excluding deprecated questions for selected release."
    )
    args = parser.parse_args()

    valid_livebench_releases = set(['2024-07-26', '2024-06-24', '2024-08-31'])

    if args.livebench_release_option not in valid_livebench_releases:
        raise ValueError(f"Bad release {args.livebench_release_option}.")
        
    release_set = set([
        r for r in valid_livebench_releases if r <= args.livebench_release_option
    ])

    if args.question_source == "huggingface":
        categories, tasks = get_categories_tasks(args.bench_name)

        for category_name, task_names in tasks.items():
            for task_name in task_names:
                questions = load_questions(categories[category_name], release_set, task_name, args.question_begin, args.question_end)
                if args.first_n:
                    questions = questions[: args.first_n]
                questions = questions[args.question_begin:args.question_end]

                questions = [
                    q for q in questions if q['livebench_removal_date'] == "" or q['livebench_removal_date'] > args.livebench_release_option
                ]


                task_full_name = f"{LIVE_BENCH_DATA_SUPER_PATH}/{category_name}/{task_name}"
                output_file = f"data/{task_full_name}/model_judgment/ground_truth_judgment.jsonl"

                answer_dir = f"data/{task_full_name}/model_answer/"

                gen_judgments(
                    parallel=args.parallel,
                    questions=questions,
                    output_file=output_file,
                    answer_dir=answer_dir,
                    model_list=args.model_list,
                    remove_existing_file=args.remove_existing_file,
                    bench_name=task_full_name,
                )


    elif args.question_source == "jsonl":
        list_of_question_files = []
        original_question_file = f"data/{args.bench_name}/question.jsonl"
        if os.path.exists(original_question_file):
            list_of_question_files = [original_question_file]
        else:
            list_of_question_files = glob.glob(f"data/{args.bench_name}/**/question.jsonl", recursive=True)

        for question_file in list_of_question_files:
            print(question_file)
            questions = load_questions_jsonl(question_file, release_set, args.question_begin, args.question_end)
            if args.first_n:
                questions = questions[: args.first_n]
            questions = questions[args.question_begin:args.question_end]

            questions = [
                q for q in questions if q['livebench_removal_date'] == "" or q['livebench_removal_date'] > args.livebench_release_option
            ]

            bench_name = os.path.dirname(question_file).replace("data/","")

            output_file = f"data/{bench_name}/model_judgment/ground_truth_judgment.jsonl"
            answer_dir = f"data/{bench_name}/model_answer/"
            if len(questions) > 0:
                gen_judgments(
                    parallel=args.parallel,
                    questions=questions,
                    output_file=output_file,
                    answer_dir=answer_dir,
                    model_list=args.model_list,
                    remove_existing_file=args.remove_existing_file,
                    bench_name=bench_name,
                )

    else:
        raise ValueError(f"Bad question source {args.question_source}.")
    