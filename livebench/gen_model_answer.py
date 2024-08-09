"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
import random
import time
import glob

import shortuuid
import torch
from tqdm import tqdm

from livebench.common import (
    reorg_answer_file,
    get_categories_tasks,
    get_hf_dataset,
    get_tasks_from_hf_category,
    load_questions,
    load_questions_jsonl,
    LIVE_BENCH_DATA_SUPER_PATH,
)
from livebench.model import load_model, get_conversation_template
from fastchat.utils import str_to_torch_dtype


def run_eval(
    model_path,
    model_id,
    questions,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    dtype,
    revision,
):
    random.shuffle(questions)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model_path,
                model_id,
                questions[i : i + chunk_size],
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                dtype=dtype,
                revision=revision,
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
    model_path,
    model_id,
    questions,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    max_gpu_memory,
    dtype,
    revision,
):
    model, tokenizer = load_model(
        model_path,
        revision=revision,
        device="cuda",
        num_gpus=num_gpus_per_model,
        max_gpu_memory=max_gpu_memory,
        dtype=dtype,
        load_8bit=False,
        cpu_offloading=False,
        debug=False,
    )

    for question, answer_file in tqdm(questions):
        temperature = 0.0

        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            conv = get_conversation_template(model_id)
            turns = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer([prompt]).input_ids

                if temperature < 1e-4:
                    do_sample = False
                else:
                    do_sample = True

                # some models may error out when generating long outputs
                print('starting question', qs[:50])
                try:
                    from transformers.generation.streamers import TextStreamer
                    output_ids = model.generate(
                        torch.as_tensor(input_ids).cuda(),
                        do_sample=do_sample,
                        temperature=temperature,
                        max_new_tokens=max_new_token,
                        streamer=TextStreamer(tokenizer)
                    )
                    if model.config.is_encoder_decoder:
                        output_ids = output_ids[0]
                    else:
                        output_ids = output_ids[0][len(input_ids[0]) :]

                    # be consistent with the template's stop_token_ids
                    if conv.stop_token_ids:
                        stop_token_ids_index = [
                            i
                            for i, id in enumerate(output_ids)
                            if id in conv.stop_token_ids
                        ]
                        if len(stop_token_ids_index) > 0:
                            output_ids = output_ids[: stop_token_ids_index[0]]

                    output = tokenizer.decode(
                        output_ids,
                        spaces_between_special_tokens=False,
                    )
                    if conv.stop_str and isinstance(conv.stop_str, list):
                        stop_str_indices = sorted(
                            [
                                output.find(stop_str)
                                for stop_str in conv.stop_str
                                if output.find(stop_str) > 0
                            ]
                        )
                        if len(stop_str_indices) > 0:
                            output = output[: stop_str_indices[0]]
                    elif conv.stop_str and output.find(conv.stop_str) > 0:
                        output = output[: output.find(conv.stop_str)]

                    for special_token in tokenizer.special_tokens_map.values():
                        if isinstance(special_token, list):
                            for special_tok in special_token:
                                output = output.replace(special_tok, "")
                        else:
                            output = output.replace(special_token, "")
                    output = output.strip()

                    if conv.name == "xgen" and output.startswith("Assistant:"):
                        output = output.replace("Assistant:", "", 1).strip()
                except RuntimeError as e:
                    print("ERROR question ID: ", question["question_id"])
                    output = "ERROR"

                conv.update_last_message(output)
                turns.append(output)

            choices.append({"index": i, "turns": turns})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--model-id", type=str, required=True, help="A custom name for the model."
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="live_bench",
        help="The name of the benchmark question set.",
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
        "--max-new-token",
        type=int,
        default=4096,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.",
        default=None,
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The model revision to load.",
    )
    parser.add_argument(
        "--question-source", type=str, default="huggingface", help="The source of the questions. 'huggingface' will draw questions from huggingface. 'jsonl' will use local jsonl files to permit tweaking or writing custom questions."
    )
    parser.add_argument(
        "--livebench-releases", type=str, nargs='+', default=['2024-07-26', '2024-06-24'], help="livebench releases to use. Provide a list of options, current options are {'2024-07-26' (july update), '2024-06-24' (original release)}. Providing all of these will run all questions."
    )
    args = parser.parse_args()

    release_set = set(args.livebench_releases)
    for r in release_set:
        if r not in set(['2024-07-26', '2024-06-24']):
            raise ValueError(f"Bad release {r}.")

    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    questions_all = []
    answer_files  = []

    if args.question_source == "huggingface":
        categories, tasks = get_categories_tasks(args.bench_name)

        for category_name, task_names in tasks.items():
            for task_name in task_names:
                questions = load_questions(categories[category_name], release_set, task_name, args.question_begin, args.question_end)

                task_full_name = f"{LIVE_BENCH_DATA_SUPER_PATH}/{category_name}/{task_name}"
                answer_file = f"data/{task_full_name}/model_answer/{args.model_id}.jsonl"

                questions_all.extend(
                    [
                        (q, answer_file)
                        for q in questions
                    ]
                )

                answer_files.append(answer_file)

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

            bench_name = os.path.dirname(question_file).replace("data/","")
            answer_file = f"data/{bench_name}/model_answer/{args.model_id}.jsonl"

            questions_all.extend(
                [
                    (q, answer_file)
                    for q in questions
                ]
            )

            if len(questions) > 0:
                answer_files.append(answer_file)

    else:
        raise ValueError(f"Bad question source {args.question_source}.")

    run_eval(
        model_path=args.model_path,
        model_id=args.model_id,
        questions=questions_all,
        max_new_token=args.max_new_token,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        max_gpu_memory=args.max_gpu_memory,
        dtype=str_to_torch_dtype(args.dtype),
        revision=args.revision,
    )

    for answer_file in answer_files:
        reorg_answer_file(answer_file)
