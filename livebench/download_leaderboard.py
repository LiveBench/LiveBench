import json
import os
from tqdm import tqdm

from livebench.common import (
    get_categories_tasks,
    load_answers_judgments,
    LIVE_BENCH_DATA_SUPER_PATH,
)

model_answer, model_judgment = load_answers_judgments()

for dir_name, file_name, dataset in [
    ('leaderboard_model_answer', 'all_model_answers.jsonl', model_answer),
    ('leaderboard_model_judgment', 'all_model_judgments.jsonl', model_judgment)
]:

    categories, tasks = get_categories_tasks(LIVE_BENCH_DATA_SUPER_PATH)

    for category_name, task_names in tqdm(tasks.items()):
        rows = dataset[category_name]
        category_path = f"data/{LIVE_BENCH_DATA_SUPER_PATH}/{category_name}/{dir_name}"
        file_path = f"{category_path}/{file_name}"

        os.makedirs(category_path, exist_ok=True)
        with open(file_path, 'w') as f:
            f.writelines([json.dumps(row, default=str) + '\n' for row in rows])

