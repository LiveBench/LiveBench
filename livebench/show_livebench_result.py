"""
Usage:
python3 show_livebench_result.py
"""
import argparse
import pandas as pd
import glob
import os
import re
import numpy as np

from livebench.common import (
    LIVE_BENCH_RELEASES,
    get_categories_tasks,
    load_questions,
    load_questions_jsonl
)
from livebench.model import get_model_config

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', None)


def calculate_usage(args, df, questions_all):
    """Calculate average token usage for all answers by task and category."""
    print("Calculating token usage for all answers...")
    
    # Get the set of valid question IDs
    valid_question_ids = {q['question_id'] for q in questions_all}
    
    # Get model list to filter by if provided
    model_filter = None
    if args.model_list is not None:
        model_filter = {get_model_config(x).display_name.lower() for x in args.model_list}
        print(f"Filtering token usage for models: {', '.join(sorted(model_filter))}")
    
    # Load model answer files
    model_answers = {}
    for bench in args.bench_name:
        # Find all model answer files without filtering by model name
        answer_files = glob.glob(f"data/{bench}/**/model_answer/*.jsonl", recursive=True)
        
        for answer_file in answer_files:
            # Load the answer file
            if os.path.exists(answer_file):
                answers = pd.read_json(answer_file, lines=True)
                
                # Skip if empty or doesn't have model_id column
                if len(answers) == 0 or 'model_id' not in answers.columns:
                    continue
                
                # Filter to only include valid question IDs
                answers = answers[answers['question_id'].isin(valid_question_ids)]
                
                # Skip if empty after filtering
                if len(answers) == 0:
                    continue
                
                # Group answers by model_id
                grouped_answers = answers.groupby('model_id')
                
                # Process each model group
                for model_id, model_group in grouped_answers:
                    # Check if this model_id matches any model in our correct answers
                    if not isinstance(model_id, str):
                        continue
                    
                    # Skip if we're filtering by model list and this model isn't in it
                    if model_filter is not None and model_id.lower() not in model_filter:
                        continue
                    
                    matching_models = [m for m in set(df["model"]) if isinstance(m, str) and m.lower() == model_id.lower()]
                    
                    for model in matching_models:
                        if model not in model_answers:
                            model_answers[model] = model_group
                        else:
                            model_answers[model] = pd.concat([model_answers[model], model_group], ignore_index=True)
    
    # Create dataframe for token usage
    usage_data = []
    
    # Process each model
    for model, answers_df in model_answers.items():
        # Check if total_output_tokens exists in the dataframe
        if 'total_output_tokens' not in answers_df.columns:
            print(f"Model {model} missing total_output_tokens data")
            continue
            
        # Filter answers to only include those with token data and where total_output_tokens is not -1
        valid_answers = answers_df.dropna(subset=['total_output_tokens'])
        valid_answers = valid_answers[valid_answers['total_output_tokens'] != -1]
        
        # Get all answers for this model
        model_all = df[df["model"] == model]
        
        # Process all answers
        for _, judgment in model_all.iterrows():
            question_id = judgment["question_id"]
            
            # Skip if question_id not in valid_question_ids
            if question_id not in valid_question_ids:
                continue
                
            matching_answer = valid_answers[valid_answers["question_id"] == question_id]
            
            if len(matching_answer) == 0:
                continue
                
            # Add to usage data
            usage_data.append({
                "model": model,
                "question_id": question_id,
                "task": judgment["task"],
                "category": judgment["category"],
                "total_output_tokens": matching_answer.iloc[0]["total_output_tokens"]
            })
    
    if not usage_data:
        print("No token usage data found.")
        return
        
    # Create dataframe from collected data
    usage_df = pd.DataFrame(usage_data)
    
    # Calculate average by task
    task_usage = usage_df.groupby(["model", "task"])["total_output_tokens"].mean().reset_index()
    task_pivot = pd.pivot_table(task_usage, index=['model'], values="total_output_tokens", columns=["task"])
    
    # Calculate average by category
    category_usage = usage_df.groupby(["model", "task", "category"])["total_output_tokens"].mean().reset_index()
    
    # Get tasks per category from the raw df
    tasks_by_category = {}
    for _, row in df.iterrows():
        category = row["category"]
        task = row["task"]
        if category not in tasks_by_category:
            tasks_by_category[category] = set()
        tasks_by_category[category].add(task)
    
    # Debug print to check tasks_by_category
    print("\nTasks by category (from raw df):")
    for category, tasks in tasks_by_category.items():
        print(f"{category}: {sorted(tasks)}")
    
    # Calculate averages
    category_pivot = pd.pivot_table(
        category_usage, 
        index=['model'], 
        values="total_output_tokens", 
        columns=["category"]
    )
    
    # Check each model and category - if not all tasks in category have data, mark as NaN
    for model in category_pivot.index:
        for category, tasks in tasks_by_category.items():
            if category in category_pivot.columns:
                # Get tasks for this model and category in all answers
                tasks_with_data = set(usage_df[(usage_df["model"] == model) & 
                                                    (usage_df["category"] == category)]["task"])
                # If any task is missing, set to NaN
                if not tasks.issubset(tasks_with_data):
                    category_pivot.at[model, category] = np.nan
    
    # Calculate averages
    avg_by_model = {}
    
    # List of all categories
    all_categories = list(category_pivot.columns)
    
    for model in category_pivot.index:
        # Check if any category has a NaN value
        has_missing_category = any(pd.isna(category_pivot.at[model, cat]) for cat in all_categories if cat in category_pivot.columns)
        
        # Only calculate average if all categories have data
        if not has_missing_category:
            values = [category_pivot.at[model, cat] for cat in all_categories if cat in category_pivot.columns]
            avg_by_model[model] = sum(values) / len(values)
    
    # Add average column
    category_pivot['average'] = pd.Series({
        model: avg_by_model.get(model, 0) 
        for model in category_pivot.index
        if model in avg_by_model
    })
    
    # Sort by average
    # Models with complete data come first (sorted by their averages)
    # Then models with incomplete data (sorted alphabetically)
    models_with_average = [model for model in category_pivot.index if model in avg_by_model]
    models_without_average = [model for model in category_pivot.index if model not in avg_by_model]
    
    sorted_models_with_average = sorted(
        models_with_average,
        key=lambda model: avg_by_model.get(model, 0),
        reverse=True
    )
    
    sorted_models = sorted_models_with_average + sorted(models_without_average)
    category_pivot = category_pivot.reindex(sorted_models)
    
    # Move average to first column
    first_col = category_pivot.pop('average')
    category_pivot.insert(0, 'average', first_col)
    
    # Format values as decimals for better readability
    category_pivot = category_pivot.round(1)
    task_pivot = task_pivot.round(1)
    
    # Save to CSV
    task_pivot.to_csv('task_usage.csv')
    category_pivot.to_csv('group_usage.csv')
    
    # Print results
    print("\n########## Token Usage by Task ##########")
    with pd.option_context('display.max_rows', None):
        print(task_pivot.sort_index())
        
    print("\n########## Token Usage by Category ##########")
    with pd.option_context('display.max_rows', None):
        print(category_pivot)


def display_result_single(args):

    if args.livebench_release_option not in LIVE_BENCH_RELEASES:
        raise ValueError(f"Bad release {args.livebench_release_option}.")
    print(f"Using release {args.livebench_release_option}")
    release_set = set([
        r for r in LIVE_BENCH_RELEASES if r <= args.livebench_release_option
    ])

    input_files = []
    for bench in args.bench_name:
        files = (
            glob.glob(f"data/{bench}/**/model_judgment/ground_truth_judgment.jsonl", recursive=True)
        )
        if args.prompt_testing:
            files += (
                glob.glob(f"prompt_testing/{bench}/**/model_judgment/ground_truth_judgment.jsonl", recursive=True)
            )
        input_files += files
    
    questions_all = []
    if args.question_source == "huggingface":
        categories = {}
        tasks = {}
        for bench in args.bench_name:
            hf_bench = bench
            # check if bench ends with _{i} for some number i
            number_match = re.match(r'(.*)_\d+$', hf_bench)
            if number_match:
                hf_bench = number_match.group(1)
            bench_cats, bench_tasks = get_categories_tasks(hf_bench)
            categories.update(bench_cats)
            for k, v in bench_tasks.items():
                if k in tasks and isinstance(tasks[k], list):
                    tasks[k].extend(v)
                else:
                    tasks[k] = v
        print(tasks)

        for category_name, task_names in tasks.items():
            for task_name in task_names:
                questions = load_questions(categories[category_name], release_set, args.livebench_release_option, task_name, None)
                questions_all.extend(questions)
    elif args.question_source == "jsonl":
        for bench in args.bench_name:
            list_of_question_files = []
            original_question_file = f"data/{bench}/question.jsonl"
            if os.path.exists(original_question_file):
                list_of_question_files = [original_question_file]
            else:
                list_of_question_files = glob.glob(f"data/{bench}/**/question.jsonl", recursive=True)

            for question_file in list_of_question_files:
                print(question_file)
                questions = load_questions_jsonl(question_file, release_set, args.livebench_release_option, None)
                questions_all.extend(questions)

    print('loaded ', len(questions_all), ' questions')
    question_id_set = set([q['question_id'] for q in questions_all])

    df_all = pd.concat((pd.read_json(f, lines=True) for f in input_files), ignore_index=True)
    df = df_all[["model", "score", "task", "category","question_id"]]
    df = df[df["score"] != -1]
    df = df[df['question_id'].isin(question_id_set)]
    df['model'] = df['model'].str.lower()
    df["score"] *= 100
    
    if args.model_list is not None:
        model_list = [get_model_config(x).display_name for x in args.model_list]
        df = df[df["model"].isin([x.lower() for x in model_list])]
        model_list_to_check = model_list
    else:
        model_list_to_check = set(df["model"])
    for model in model_list_to_check:
        df_model = df[df["model"] == model]

        missing_question_ids = set([q['question_id'] for q in questions_all]) - set(df_model['question_id'])
        
        if len(missing_question_ids) > 0 and not args.ignore_missing_judgments:
            if args.verbose:
                print('removing model', model, "has missing", len(questions_all) - len(df_model), "judgments - has ", len(df_model))
                if len(missing_question_ids) < 10:
                    print('missing ids', missing_question_ids)
                missing_questions = [q for q in questions_all if q['question_id'] in missing_question_ids]
                missing_tasks = set([q['task'] for q in missing_questions])
                print('missing tasks', missing_tasks)
            df = df[df["model"] != model]
        elif len(missing_question_ids) > 0 and args.ignore_missing_judgments:
            questions_all = [q for q in questions_all if q['question_id'] in df_model['question_id'].values]
    
    if args.ignore_missing_judgments and len(questions_all) == 0:
        raise ValueError("No questions left after ignoring missing judgments.")
    
    if args.ignore_missing_judgments:
        print(f"{len(questions_all)} questions after removing those with missing judgments.")

    
    df = df[df['question_id'].isin([q['question_id'] for q in questions_all])]

    df.to_csv('df_raw.csv')

    print("\n########## All Tasks ##########")
    df_1 = df[["model", "score", "task"]]
    df_1 = df_1.groupby(["model", "task"]).mean()
    df_1 = pd.pivot_table(df_1, index=['model'], values = "score", columns=["task"], aggfunc="sum")
    if args.show_average_row:
        df_1.loc['average'] = df_1.mean()
    df_1 = df_1.round(3)
    df_1 = df_1.dropna(inplace=False)
    with pd.option_context('display.max_rows', None):
        print(df_1.sort_values(by="model"))
    df_1.to_csv('all_tasks.csv')

    if not args.prompt_testing:
        print("\n########## All Groups ##########")
        df_1 = df[["model", "score", "category", "task"]]
        df_1 = df_1.groupby(["model", "task", "category"]).mean().groupby(["model","category"]).mean()
        df_1 = pd.pivot_table(df_1, index=['model'], values = "score", columns=["category"], aggfunc="sum")

        df_1 = df_1.dropna(inplace=False)

        # Only show average column if there are multiple data columns and not explicitly skipped
        if not args.skip_average_column and len(df_1.columns) > 1:
            df_1['average'] = df_1.mean(axis=1)
            first_col = df_1.pop('average')
            df_1.insert(0, 'average', first_col)
            sort_by = "average"
        else:
            sort_by = df_1.columns[0] if len(df_1.columns) > 0 else None

        # Sort if we have a column to sort by
        if sort_by is not None:
            df_1 = df_1.sort_values(by=sort_by, ascending=False)

        df_1 = df_1.round(1)
        if args.show_average_row:
            df_1.loc['average'] = df_1.mean()
        with pd.option_context('display.max_rows', None):
            print(df_1)
        df_1.to_csv('all_groups.csv')


        for column in df_1.columns[1:]:
            max_value = df_1[column].max()
            df_1[column] = df_1[column].apply(lambda x: f'\\textbf{{{x}}}' if x == max_value else x)
        df_1.to_csv('latex_table.csv', sep='&', lineterminator='\\\\\n', quoting=3,escapechar=" ")
    
    if args.print_usage:
        calculate_usage(args, df, questions_all)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display benchmark results for the provided models")
    parser.add_argument("--bench-name", type=str, default=["live_bench"], nargs="+")
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
        "--livebench-release-option", 
        type=str, 
        default=max(LIVE_BENCH_RELEASES),
        choices=LIVE_BENCH_RELEASES,
        help="Livebench release to use. Provide a single date option. Will handle excluding deprecated questions for selected release."
    )
    parser.add_argument(
        "--show-average-row",
        default=False,
        help="Show the average score for each task",
        action='store_true'
    )
    parser.add_argument(
        "--ignore-missing-judgments",
        default=False,
        action='store_true',
        help="Ignore missing judgments. Scores will be calculated for only questions that have judgments for all models."
    )
    parser.add_argument(
        "--print-usage",
        default=False,
        action='store_true',
        help="Calculate and display token usage for correct answers"
    )
    parser.add_argument(
        "--verbose",
        default=False,
        help="Display debug information",
        action='store_true'
    )
    parser.add_argument(
        "--skip-average-column",
        default=False,
        help="Skip displaying the average column in results",
        action='store_true'
    )
    parser.add_argument(
        "--prompt-testing",
        default=False,
        help="Use prompt testing results in addition to normal results",
        action="store_true"
    )
    args = parser.parse_args()

    display_result_func = display_result_single

    display_result_func(args)
