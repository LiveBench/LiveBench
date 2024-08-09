import json
import os
from tqdm import tqdm

from livebench.common import (
    get_categories_tasks,
    load_questions,
    LIVE_BENCH_DATA_SUPER_PATH,
)

categories, tasks = get_categories_tasks(LIVE_BENCH_DATA_SUPER_PATH)

for category_name, task_names in tqdm(tasks.items()):
    # load questions from all livebench releases
    livebench_releases = set(['2024-07-26', '2024-06-24'])
    questions = load_questions(categories[category_name], livebench_releases=livebench_releases, task_name=None, begin=None, end=None)

    category_path = f"data/{LIVE_BENCH_DATA_SUPER_PATH}/{category_name}"
    question_file_path = f"{category_path}/question.jsonl"

    os.makedirs(category_path, exist_ok=True)
    with open(question_file_path, 'w') as f:
        f.writelines([json.dumps(example, default=str) for example in questions])