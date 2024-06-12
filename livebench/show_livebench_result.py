"""
Usage:
python3 show_livebench_result.py
"""
import argparse
import pandas as pd
import glob


def display_result_single(args):
    if args.input_file is None:
        input_files = (
            glob.glob(f"data/{args.bench_name}/**/model_judgment/ground_truth_judgment.jsonl", recursive=True)
        )
    else:
        input_files = args.input_file

    df_all = pd.concat((pd.read_json(f, lines=True) for f in input_files), ignore_index=True)
    df = df_all[["model", "score", "category", "grouping"]]
    df = df[df["score"] != -1]
    df['model'] = df['model'].str.lower()
    df["score"] *= 100

    if args.model_list is not None:
        df = df[df["model"].isin([x.lower() for x in args.model_list])]

    df.loc[df['category'].str.contains('amps_hard_'), 'category'] = 'AMPS_Hard'
    df.loc[df['category'].str.contains('aime_'), 'category'] = 'math_comp'
    df.loc[df['category'].str.contains('amc_'), 'category'] = 'math_comp'
    df.loc[df['category'] == 'smc', 'category'] = 'math_comp'
    df.loc[df['category'].isin(['usamo', 'imo']), 'category'] = 'olympiad'


    print("\n########## All Tasks ##########")
    df_1 = df[["model", "score", "category"]]
    df_1 = df_1.groupby(["model", "category"]).mean()
    df_1 = pd.pivot_table(df_1, index=['model'], values = "score", columns=["category"], aggfunc="sum")
    df_1 = df_1.round(3)
    print(df_1.sort_values(by="model"))
    df_1.to_csv('all_tasks.csv')

    print("\n########## All Groups ##########")
    df_1 = df[["model", "score", "grouping", "category"]]
    df_1 = df_1.groupby(["model", "category", "grouping"]).mean().groupby(["model","grouping"]).mean()
    df_1 = pd.pivot_table(df_1, index=['model'], values = "score", columns=["grouping"], aggfunc="sum")

    df_1['average'] = df_1.mean(axis=1)
    first_col = df_1.pop('average')
    df_1.insert(0, 'average', first_col)
    df_1 = df_1.sort_values(by="average", ascending=False)
    df_1 = df_1.round(3)
    print(df_1)
    df_1.to_csv('all_groups.csv')



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
    args = parser.parse_args()

    display_result_func = display_result_single

    display_result_func(args)
