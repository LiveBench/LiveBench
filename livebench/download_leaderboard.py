import json
import os
from tqdm import tqdm

from livebench.common import (
    get_categories_tasks,
    load_answers_judgments,
    LIVE_BENCH_DATA_SUPER_PATH,
)

model_answer, model_judgment = load_answers_judgments()

for dir_name, dataset in [
    ('model_answer', model_answer),
    ('model_judgment', model_judgment)
]:

    categories, tasks = get_categories_tasks(LIVE_BENCH_DATA_SUPER_PATH)

    for category_name, task_names in tqdm(tasks.items()):
        rows = dataset[category_name]
        for task_name in task_names:
            rows_task = [
                r for r in rows if r['task'] == task_name
            ]
            if dir_name == 'model_judgment':

                task_path = f"data/{LIVE_BENCH_DATA_SUPER_PATH}/{category_name}/{task_name}/{dir_name}"
                file_path = f"{task_path}/ground_truth_judgment.jsonl"

                os.makedirs(task_path, exist_ok=True)
                with open(file_path, 'w') as f:
                    f.writelines([json.dumps(row, default=str) + '\n' for row in rows_task])

            else:
                models = set(
                    [
                        row['model_id'] for row in rows_task
                    ]
                )

                for model in models:
                    rows_model = [
                        r for r in rows_task if r['model_id'] == model
                    ]

                    task_path = f"data/{LIVE_BENCH_DATA_SUPER_PATH}/{category_name}/{task_name}/{dir_name}"
                    file_path = f"{task_path}/{model}.jsonl"

                    os.makedirs(task_path, exist_ok=True)
                    with open(file_path, 'w') as f:
                        f.writelines([json.dumps(row, default=str) + '\n' for row in rows_model])
