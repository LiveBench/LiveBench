"""
Usage:
python3 show_livebench_result.py
"""
import argparse
import pandas as pd
import glob
import os

from livebench.common import (
    get_categories_tasks,
    load_questions,
    load_questions_jsonl
)


def display_result_single(args, update_names=True):

    release_set = set(args.livebench_releases)
    for r in release_set:
        if r not in set(['2024-07-26', '2024-06-24']):
            raise ValueError(f"Bad release {r}.")

    if args.input_file is None:
        input_files = (
            glob.glob(f"data/{args.bench_name}/**/model_judgment/ground_truth_judgment.jsonl", recursive=True)
        )
    else:
        input_files = args.input_file

    categories, tasks = get_categories_tasks(args.bench_name)

    questions_all = []
    if args.question_source == "huggingface":
        for category_name, task_names in tasks.items():
            for task_name in task_names:
                questions = load_questions(categories[category_name], release_set, task_name, None, None)
                questions_all.extend(questions)
        
    elif args.question_source == "jsonl":
        list_of_question_files = []
        original_question_file = f"data/{args.bench_name}/question.jsonl"
        if os.path.exists(original_question_file):
            list_of_question_files = [original_question_file]
        else:
            list_of_question_files = glob.glob(f"data/{args.bench_name}/**/question.jsonl", recursive=True)

        for question_file in list_of_question_files:
            print(question_file)
            questions = load_questions_jsonl(question_file, release_set, None, None)
            questions_all.extend(questions)


    question_id_set = set([q['question_id'] for q in questions_all])
    print(len(question_id_set))

    df_all = pd.concat((pd.read_json(f, lines=True) for f in input_files), ignore_index=True)
    df = df_all[["model", "score", "task", "category","question_id"]]
    df = df[df["score"] != -1]
    df = df[df['question_id'].isin(question_id_set)]
    df['model'] = df['model'].str.lower()
    df["score"] *= 100

    if update_names:
        df.loc[df['model'] == 'gemini-1.5-pro-latest', 'model'] = 'gemini-1.5-pro-api-0514'
        df.loc[df['model'] == 'gemini-1.5-flash-latest', 'model'] = 'gemini-1.5-flash-api-0514'
        df.loc[df['model'] == 'deepseek-coder', 'model'] = 'deepseek-coder-v2'
        df.loc[df['model'] == 'deepseek-chat', 'model'] = 'deepseek-chat-v2'
        df.loc[df['model'] == 'acm_rewrite_qwen2-72b-chat', 'model'] = 'smaug-qwen2-72b-instruct'
        df.loc[df['model'] == 'open-mixtral-8x22b', 'model'] = 'mixtral-8x22b-instruct-v0.1'    
        df.loc[df['model'] == 'open-mixtral-8x7b', 'model'] = 'mixtral-8x7b-instruct-v0.1'    
        df = df[df["model"] != "gpt-4-1106-preview"]
        df = df[df["model"] != "gpt-3.5-turbo-1106"]

    if args.model_list is not None:
        df = df[df["model"].isin([x.lower() for x in args.model_list])]
        model_list_to_check = args.model_list
    else:
        model_list_to_check = set(df["model"])
    for model in model_list_to_check:
        df_model = df[df["model"] == model]
        
        if len(df_model) < len(questions_all):
            raise ValueError(f'Invalid result, missing judgments (and possibly completions) for {len(questions_all) - len(df_model)} questions for model {model}.')


    print(len(df))

    print("\n########## All Tasks ##########")
    df_1 = df[["model", "score", "task"]]
    df_1 = df_1.groupby(["model", "task"]).mean()
    df_1 = pd.pivot_table(df_1, index=['model'], values = "score", columns=["task"], aggfunc="sum")
    df_1 = df_1.round(3)
    print(df_1.sort_values(by="model"))
    df_1.to_csv('all_tasks.csv')

    print("\n########## All Groups ##########")
    df_1 = df[["model", "score", "category", "task"]]
    df_1 = df_1.groupby(["model", "task", "category"]).mean().groupby(["model","category"]).mean()
    df_1 = pd.pivot_table(df_1, index=['model'], values = "score", columns=["category"], aggfunc="sum")

    df_1['average'] = df_1.mean(axis=1)
    first_col = df_1.pop('average')
    df_1.insert(0, 'average', first_col)
    df_1 = df_1.sort_values(by="average", ascending=False)
    df_1 = df_1.round(1)
    print(df_1)
    df_1.to_csv('all_groups.csv')


    for column in df_1.columns[1:]:
        max_value = df_1[column].max()
        df_1[column] = df_1[column].apply(lambda x: f'\\textbf{{{x}}}' if x == max_value else x)
    df_1.to_csv('latex_table.csv', sep='&', lineterminator='\\\\\n', quoting=3,escapechar=" ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-name", type=str, default="live_bench")
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--baseline-model", type=str, default="gpt-3.5-turbo")
    parser.add_argument(
        "--questions-equivalent", 
        action=argparse.BooleanOptionalAction,
        help="""Use this argument to treat all questions with the same weight. 
        If this argument is not included, all categories have the same weight."""
    )
    parser.add_argument(
        "--model-list",
        type=str,
        nargs="+",
        default=None,
        help="A list of models to be evaluated",
    )
    parser.add_argument(
        "--question-source", type=str, default="huggingface", help="The source of the questions. 'huggingface' will draw questions from huggingface. 'jsonl' will use local jsonl files to permit tweaking or writing custom questions."
    )
    parser.add_argument(
        "--livebench-releases", type=str, nargs='+', default=['2024-07-26', '2024-06-24'], help="livebench releases to use. Provide a list of options, current options are {'2024-07-26' (july update), '2024-06-24' (original release)}. Providing all of these will run all questions."
    )
    args = parser.parse_args()

    display_result_func = display_result_single

    display_result_func(args)
