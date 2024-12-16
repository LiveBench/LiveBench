import pandas as pd
from io import StringIO
import re


def read_df_func(df_type, df_str):
    if df_type == "json":
        try:
            return pd.read_json(StringIO(df_str), orient="index", encoding="utf-8")
        except:
            pass
        try:
            return pd.read_json(StringIO(df_str), orient="records", lines=False, encoding="utf-8")
        except:
            pass
        try:
            return pd.read_json(StringIO(df_str), orient="records", lines=True, encoding="utf-8")
        except:
            pass
        try:
            return pd.read_json(StringIO(df_str), orient="table", encoding="utf-8")
        except:
            pass
        return pd.read_json(StringIO(df_str), orient="values", encoding="utf-8")
    elif df_type == "jsonl":
        return pd.read_json(StringIO(df_str), orient="records", lines=True, encoding="utf-8")
    elif df_type == "html":
        return pd.concat(pd.read_html(StringIO(df_str), encoding="utf-8"), axis=0)
    elif df_type == "csv":
        return pd.read_csv(StringIO(df_str), encoding="utf-8")
    elif df_type == "markdown":
        return pd.read_table(StringIO(df_str), sep="|", header=0, index_col=1, skipinitialspace=True)
    elif df_type == "tsv":
        return pd.read_csv(StringIO(df_str), sep='\t', encoding="utf-8")

def clean_llm_output(s):
    pattern_json = r'```json\n(.*?)```'
    matches = re.findall(pattern_json, s, re.DOTALL)
    if len(matches) > 0:
        return matches[-1].strip()
    pattern_a = r'^```.*\n'
    s = re.sub(pattern_a, "", s)
    # re.findall(pattern, text, re.MULTILINE)
    return s.replace("```", "").strip()

def remove_initial_phrase(text):
    # remove phrases like "Here is the updated table:" , "Here is the table in a new format:"
    pattern = r'^\s*(Here|Input)\b.*?\b(format|table)\s*[:)]\s*'
    modified_text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    return modified_text.strip()

def table_process_results(input_command: str, ground_truth: str, llm_answer: str, debug=False) -> int:
    input_format = input_command.split("Please convert the Input Table from ")[1].split(" format")[0].lower()
    output_format = input_command.split("Please convert the Input Table from ")[1].split("format to ")[1].split(" format")[0].lower()
    gt_df = read_df_func(output_format, ground_truth)
    try:
        gt_df = read_df_func(output_format, ground_truth)
    except:
        print('Error when reading the ground-truth table')
        return "err"
    llm_clean = clean_llm_output(llm_answer)
    # if there's an initial phrase before the table, remove it and try to score again
    llm_clean = remove_initial_phrase(llm_clean)
    # first check the raw LLM output
    llm_df = None
    try:
        llm_df = read_df_func(output_format, llm_clean)
    except:
        print('Could not read the LLM output')
        print('GROUND TRUTH\n', ground_truth)
        print('END OF OUTPUT\n', llm_answer[-min(3000, len(llm_answer)):])
        return 0
    score = check_table_reformat(output_format, llm_df, gt_df, debug)
    if debug and score == 0:
        print('INCORRECT')
        print('GROUND TRUTH\n', gt_df)
        print('LLM DF\n', llm_df)
        print('LLM ANSWER\n', llm_clean)
    return score

def check_table_reformat(output_format, llm_df, gt_df, debug=False):
    try:
        gt_df.columns = [s.lower().strip() for s in gt_df.columns]
        llm_df.columns = [s.lower().strip() for s in llm_df.columns]
        assert len(llm_df) == len(gt_df), f"DataFrame Length does not match, {len(llm_df)} (LLM) vs {len(gt_df)} (Ground Truth)"
        assert list(sorted(llm_df.columns)) == list(sorted(gt_df.columns)), f"Columns do not match:\n{sorted(llm_df.columns)} (LLM)\n{sorted(gt_df.columns)} (Ground Truth)"
        # for test_col in llm_df.columns:
        #     assert sorted(llm_df[test_col].tolist()) == sorted(gt_df[test_col].tolist()), f"Column content {test_col} does not match"
        for i in range(len(llm_df)):
            stripped_gt_df = {key.strip(): value.strip() if isinstance(value, str) else value for key, value in gt_df.iloc[i].to_dict().items()} # some ground truth values are strings with trailing spaces
            assert llm_df.iloc[i].to_dict() == gt_df.iloc[i].to_dict() or llm_df.iloc[i].to_dict() == stripped_gt_df, f"Row {i} does not match:\n{llm_df.iloc[i].to_dict()} (LLM)\n{stripped_gt_df} (Ground Truth)"
    except Exception as e:
        if debug:
            print(e)
        return 0
    return 1
