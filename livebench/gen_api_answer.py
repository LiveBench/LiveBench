"""Generate answers with GPT-4

Usage:
python3 gen_api_answer.py --model gpt-3.5-turbo
"""

import argparse
import json
import os
import time
import concurrent.futures
import glob
import shortuuid
import tqdm
import subprocess
import sys
from datetime import datetime

from livebench.model.api_model_config import APIKwargs, AgentConfig
from livebench.agentic_code_runner.minisweagent.run_inference import run_agentic_coding_inference

from livebench.common import (
    LIVE_BENCH_RELEASES,
    reorg_answer_file,
    get_categories_tasks,
    load_questions,
    load_questions_jsonl,
    LIVE_BENCH_DATA_SUPER_PATH,
    filter_questions,
    check_agentic_coding_requirements
)

from livebench.model import ModelConfig, get_model_config, get_api_function


def get_answer(
    question: dict,
    model_api_name: str,
    provider: str,
    force_temperature: float | None,
    max_tokens: int,
    num_choices: int,
    model_config: ModelConfig,
    api_kwargs: dict[str, str] | None = None,
    api_dict: dict[str, str] | None = None,
    stream: bool = False,
    answer_file: str | None = None,
    model_display_name_override: str | None = None,
    model_provider_override: str | None = None,

):
    """
    Perform inference for a single question.

    Args:
        question: At minimum, a dictionary with a key 'turns' that maps to a list of messages in the conversation, the last of which should ask the question.
        model: The API name for the model (e.g. gpt-4o-mini or claude-3-5-sonnet-20240620)
        model_display_name_override: The display name for the model (e.g. gpt-4o-mini or claude-3-5-sonnet-20240620)
        num_choices: The number of model outputs to generate for each question
        max_tokens: The maximum number of tokens for each model response
        answer_file: The path to the file in which to write answers
        api_dict: A dictionary specifying the base API URL and key for model requests
    """

    assert (
        force_temperature is not None and "required_temperature" in question.keys()
    ) is False
    if force_temperature is not None:
        temperature = force_temperature
    elif "required_temperature" in question.keys():
        temperature = question["required_temperature"]
    else:
        temperature = 0

    metadata = {}

    
    choices = []
    total_num_tokens = 0
    for i in range(num_choices):
        messages = []

        if 'system_prompt' in question:
            messages.append({"role": "system", "content": question['system_prompt']})

        turns = []
        for j in range(len(question["turns"])):
            prompt = question["turns"][j]
            if model_config.prompt_prefix:
                prompt = model_config.prompt_prefix + "\n" + prompt
            messages.append({"role": "user", "content": prompt})

            res = get_api_function(provider)(
                model=model_api_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                model_api_kwargs=api_kwargs,
                api_dict=api_dict,
                stream=stream
            )

            if len(res) == 3:
                output, num_tokens, m = res
            else:
                output, num_tokens = res
                m = None

            if m is not None:
                metadata.update(m)

            messages.append({"role": "assistant", "content": output})
            turns.append(output)
            total_num_tokens += num_tokens

        choices.append({"index": i, "turns": turns})

    # Dump answers
    ans = {
        "question_id": question["question_id"],
        "answer_id": shortuuid.uuid(),
        "model_id": model_display_name_override if model_display_name_override else model_config.display_name.lower(),
        "choices": choices,
        "tstamp": time.time(),
        "total_output_tokens": total_num_tokens,
        "api_info": {
            "provider": provider if provider != 'local' else api_dict['api_base'],
            "api_name": model_api_name,
            "api_kwargs": api_kwargs,
            **metadata
        }
    }

    if answer_file is not None:
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(answer_file, "a") as fout:
            fout.write(json.dumps(ans) + "\n")


def setup_model(model_config: ModelConfig, api_dict: dict[str, str] | None = None, model_provider_override: str | None = None) -> tuple[str, APIKwargs | None, str, dict[str, str] | None]:
    
    provider = model_provider_override

    if provider is None:
        provider = model_config.default_provider
    if provider is None:
        provider = list(model_config.api_name.keys())[0] if len(model_config.api_name.keys()) > 0 else None

    if provider is None and api_dict is not None:
        assert 'api_base' in api_dict, "Missing API base for model"
        provider = 'local'

    if provider is None:
        raise ValueError(f"Missing provider for model {model_config.display_name}")
    
    if 'https' in provider:
        # provider name is a base URL
        if api_dict is None:
            api_dict = {
                'api_base': provider,   
            }
            if model_config.api_keys and os.environ.get(model_config.api_keys[provider]):
                api_dict['api_key'] = os.environ.get(model_config.api_keys[provider])
    

    api_kwargs = None
    if model_config.api_kwargs:
        api_kwargs = {}
        if model_config.api_kwargs.get('default'):
            # pull default kwargs for model
            api_kwargs = model_config.api_kwargs['default']
        if provider in model_config.api_kwargs:
            # update with provider-specific kwargs
            api_kwargs.update(model_config.api_kwargs[provider])

    if provider == 'local':
        assert api_dict is not None, "Missing API dict for local model"
        assert 'api_base' in api_dict, "Missing API base for local model"

    model_api_name = model_config.api_name[provider] if provider in model_config.api_name else model_config.display_name

    return provider, api_kwargs, model_api_name, api_dict


def run_questions(
    parallel,
    questions: list[dict],
    model_config: ModelConfig,
    num_choices: int,
    max_tokens: int,
    answer_file: str | None,
    stream: bool,
    force_temperature: float | None,
    model_provider_override: str | None,
    bench_name: str,
    model_display_name_override: str | None = None,
    api_dict: dict[str, str] | None = None,
    task_to_answer_file: dict[str, str] | None = None,
):
    """
    Perform inference on a list of questions. Output answers to answer_file.

    Args:
        questions: The list of questions.
        model: The display name for the model (e.g. gpt-4o-mini or claude-3-5-sonnet-20240620) or an alias
        num_choices: The number of model outputs to generate for each question
        max_tokens: The maximum number of tokens for each model response
        answer_file: The path to the file in which to write answers
        parallel: The number of workers to use to make concurrent API requests
        api_dict: A dictionary specifying the base API URL and key for model requests
    """

    provider, api_kwargs, model_api_name, api_dict = setup_model(model_config, api_dict, model_provider_override)

    print('Model API name: ', model_api_name)
    print('Evaluating ', len(questions), ' questions in ', bench_name, ' with model ', model_config.display_name)
   
    if 'agentic_coding' in bench_name:

        if model_config.agent_config is not None:
            if api_kwargs is None:
                api_kwargs = {}
            if 'default' in model_config.agent_config:
                api_kwargs.update(model_config.agent_config['default'])
            if provider in model_config.agent_config:
                api_kwargs.update(model_config.agent_config[provider])

        if not check_agentic_coding_requirements():
            print("Warning: litellm or docker missing, skipping agentic coding evaluation")
            return

        run_agentic_coding_inference(
            questions=questions,
            model_api_name=model_api_name,
            provider=provider,
            force_temperature=force_temperature,
            num_choices=num_choices,
            model_api_kwargs=api_kwargs,
            api_dict=api_dict,
            model_display_name_override=model_display_name_override,
            answer_file=answer_file,
            parallel=parallel,
            task_to_answer_file=task_to_answer_file
        )
    elif parallel == 1:
        for question in tqdm.tqdm(questions):
            get_answer(
                question=question,
                model_api_name=model_api_name,
                provider=provider,
                force_temperature=force_temperature,
                max_tokens=max_tokens,
                num_choices=num_choices,
                api_kwargs=api_kwargs,
                api_dict=api_dict,
                stream=stream,
                model_display_name_override=model_display_name_override,
                answer_file=answer_file,
                model_config=model_config,
            )
    else:

        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = []
            for question in questions:
                future = executor.submit(
                    get_answer,
                    question=question,
                    model_api_name=model_api_name,
                    provider=provider,
                    force_temperature=force_temperature,
                    max_tokens=max_tokens,
                    num_choices=num_choices,
                    api_kwargs=api_kwargs,
                    api_dict=api_dict,
                    stream=stream,
                    model_display_name_override=model_display_name_override,
                    answer_file=answer_file,
                    model_config=model_config,
                )
                futures.append(future)

            for future in tqdm.tqdm(
                concurrent.futures.as_completed(futures), total=len(futures)
            ):
                future.result()
    if len(questions) > 0:
        if 'agentic_coding' in bench_name and task_to_answer_file is not None:
            # For agentic_coding, reorganize each task's answer file
            for task_answer_file in task_to_answer_file.values():
                reorg_answer_file(task_answer_file)
        elif answer_file is not None:
            reorg_answer_file(answer_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate benchmark question answers using an API-based model"
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="live_bench",
        help="The name of the benchmark question set. Defaults to 'live_bench', or all tasks in the benchmark. Specify e.g. live_bench/reasoning/web_of_lies_v2 to generate only for that task.",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="If provided, will be used as the base of an openai API request, along with the environment variable LIVEBENCH_API_KEY",
    )
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--force-temperature", type=float, help="Forcibly set a sampling temperature."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="The maximum number of new generated tokens.",
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
        "--parallel", type=int, default=1, help="The number of concurrent API calls."
    )
    parser.add_argument(
        "--question-source",
        type=str,
        default="huggingface",
        help="The source of the questions. 'huggingface' will draw questions from huggingface. 'jsonl' will gather local jsonl files at data/{bench_name}/**/question.jsonl to permit tweaking or writing custom questions.",
    )
    parser.add_argument(
        "--livebench-release-option",
        type=str,
        default=max(LIVE_BENCH_RELEASES),
        choices=sorted(LIVE_BENCH_RELEASES),
        help="Livebench release to use. Provide a single date option. Will handle excluding deprecated questions for selected release.",
    )
    parser.add_argument(
        "--min-release-date",
        type=str,
        default=None,
        help="Filter questions by minimum release date (e.g., YYYY-MM-DD)",
    )
    parser.add_argument(
        "--question-id",
        type=str,
        default=None,
        nargs="+",
        help="A list of question ids to generate answers for.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Do not generate answers for questions that have already been generated, unless they were errors and --retry-failures is set."
    )
    parser.add_argument(
        "--model-display-name",
        type=str,
        default=None,
        help="Optional display name of the model. If not provided, will be inferred from --model.",
    )
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        default=False,
        help="Retry generating answers for questions that have failed in the past.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        default=False,
        help="Stream responses, for models that support streaming"
    )
    parser.add_argument(
        "--model-provider-override",
        type=str,
        default=None,
        help="Override the provider for the model. If not provided, will be inferred from --model.",
    )
    args = parser.parse_args()


    if args.livebench_release_option not in LIVE_BENCH_RELEASES:
        raise ValueError(f"Bad release {args.livebench_release_option}.")

    release_set = set(
        [r for r in LIVE_BENCH_RELEASES if r <= args.livebench_release_option]
    )

    if args.api_base is not None:
        # use manually-specified model API

        api_key = os.environ.get("LIVEBENCH_API_KEY", "EMPTY")

        api_dict = {
            "api_key": api_key,
            "api_base": args.api_base,
        }
    elif os.environ.get("LIVEBENCH_API_KEY") is not None:
        api_dict = {
            "api_key": os.environ.get("LIVEBENCH_API_KEY"),
        }
    else:
        api_dict = None

    model_config = get_model_config(args.model)
    model_display_name = args.model_display_name if args.model_display_name else model_config.display_name

    if args.question_source == "huggingface":
        categories, tasks = get_categories_tasks(args.bench_name)

        # Check if this is an agentic_coding benchmark
        is_agentic_coding = 'agentic_coding' in args.bench_name
        
        if is_agentic_coding:
            # For agentic_coding, group all questions together but maintain separate answer files
            all_questions = []
            task_to_answer_file = {}
            
            for category_name, task_names in tasks.items():
                for task_name in task_names:
                    questions = load_questions(
                        categories[category_name],
                        release_set,
                        args.livebench_release_option,
                        task_name,
                        args.question_id,
                        args.min_release_date
                    )

                    questions = questions[args.question_begin:args.question_end]

                    task_full_name = (
                        f"{LIVE_BENCH_DATA_SUPER_PATH}/{category_name}/{task_name}"
                    )
                    answer_file = (
                        f"data/{task_full_name}/model_answer/{model_display_name.lower()}.jsonl"
                    )

                    questions = filter_questions(questions, answer_file, args.resume, args.retry_failures)
                    
                    # Add task information to each question
                    for question in questions:
                        question['task'] = task_name
                    
                    all_questions.extend(questions)
                    task_to_answer_file[task_name] = answer_file
                    
                    print(f"Questions from {task_full_name}")
                    print(f"Output to {answer_file}")
            
            if all_questions:
                print(f"Running {len(all_questions)} agentic_coding questions together")
                run_questions(
                    parallel=args.parallel,
                    questions=all_questions,
                    model_config=model_config,
                    num_choices=args.num_choices,
                    max_tokens=args.max_tokens,
                    answer_file=None,  # Will use task_to_answer_file mapping instead
                    api_dict=api_dict,
                    stream=args.stream,
                    force_temperature=args.force_temperature,
                    model_provider_override=args.model_provider_override,
                    model_display_name_override=model_display_name,
                    bench_name=args.bench_name,
                    task_to_answer_file=task_to_answer_file
                )
        else:
            # For non-agentic_coding, process each task separately as before
            for category_name, task_names in tasks.items():
                for task_name in task_names:
                    questions = load_questions(
                        categories[category_name],
                        release_set,
                        args.livebench_release_option,
                        task_name,
                        args.question_id,
                        args.min_release_date
                    )

                    questions = questions[args.question_begin:args.question_end]

                    task_full_name = (
                        f"{LIVE_BENCH_DATA_SUPER_PATH}/{category_name}/{task_name}"
                    )
                    answer_file = (
                        f"data/{task_full_name}/model_answer/{model_display_name.lower()}.jsonl"
                    )

                    questions = filter_questions(questions, answer_file, args.resume, args.retry_failures)

                    print(f"Questions from {task_full_name}")
                    print(f"Output to {answer_file}")

                    run_questions(
                        parallel=args.parallel,
                        questions=questions,
                        model_config=model_config,
                        num_choices=args.num_choices,
                        max_tokens=args.max_tokens,
                        answer_file=answer_file,
                        api_dict=api_dict,
                        stream=args.stream,
                        force_temperature=args.force_temperature,
                        model_provider_override=args.model_provider_override,
                        model_display_name_override=model_display_name,
                        bench_name=task_full_name
                    )

    elif args.question_source == "jsonl":
        # use locally-provided questions

        list_of_question_files = []
        original_question_file = f"data/{args.bench_name}/question.jsonl"
        if os.path.exists(original_question_file):
            # if one specific file for bench_name exists, use it (e.g. if bench_name = live_bench/math/AMPS_Hard)
            list_of_question_files = [original_question_file]
        else:
            # gather all question files for bench_name (e.g. if bench_name = live_bench/math)
            list_of_question_files = glob.glob(
                f"data/{args.bench_name}/**/question.jsonl", recursive=True
            )

        # Check if this is an agentic_coding benchmark
        is_agentic_coding = 'agentic_coding' in args.bench_name
        
        if is_agentic_coding:
            # For agentic_coding, group all questions together but maintain separate answer files
            all_questions = []
            task_to_answer_file = {}
            
            for question_file in list_of_question_files:
                print(question_file)
                questions = load_questions_jsonl(
                    question_file, release_set, args.livebench_release_option, args.question_id, args.min_release_date
                )
                
                questions = questions[args.question_begin:args.question_end]

                bench_name = os.path.dirname(question_file).replace("data/", "")
                answer_file = f"data/{bench_name}/model_answer/{model_display_name.lower()}.jsonl"

                questions = filter_questions(questions, answer_file, args.resume, args.retry_failures)
                
                # Extract task name from bench_name (assuming format like "live_bench/agentic_coding/task_name")
                task_name = bench_name.split('/')[-1] if '/' in bench_name else bench_name
                
                # Add task information to each question
                for question in questions:
                    question['task'] = task_name
                
                all_questions.extend(questions)
                task_to_answer_file[task_name] = answer_file
                        
                print(f"Questions from {question_file}")
                print(f"Output to {answer_file}")
            
            if all_questions:
                print(f"Running {len(all_questions)} agentic_coding questions together")
                run_questions(
                    parallel=args.parallel,
                    questions=all_questions,
                    model_config=model_config,
                    model_display_name_override=model_display_name,
                    num_choices=args.num_choices,
                    max_tokens=args.max_tokens,
                    answer_file=None,  # Will use task_to_answer_file mapping instead
                    api_dict=api_dict,
                    stream=args.stream,
                    force_temperature=args.force_temperature,
                    model_provider_override=args.model_provider_override,
                    bench_name=args.bench_name,
                    task_to_answer_file=task_to_answer_file
                )
        else:
            # For non-agentic_coding, process each file separately as before
            for question_file in list_of_question_files:
                print(question_file)
                questions = load_questions_jsonl(
                    question_file, release_set, args.livebench_release_option, args.question_id, args.min_release_date
                )
                
                questions = questions[args.question_begin:args.question_end]

                bench_name = os.path.dirname(question_file).replace("data/", "")
                answer_file = f"data/{bench_name}/model_answer/{model_display_name.lower()}.jsonl"

                questions = filter_questions(questions, answer_file, args.resume, args.retry_failures)
                        
                print(f"Questions from {question_file}")
                print(f"Output to {answer_file}")

                run_questions(
                    parallel=args.parallel,
                    questions=questions,
                    model_config=model_config,
                    model_display_name_override=model_display_name,
                    num_choices=args.num_choices,
                    max_tokens=args.max_tokens,
                    answer_file=answer_file,
                    api_dict=api_dict,
                    stream=args.stream,
                    force_temperature=args.force_temperature,
                    model_provider_override=args.model_provider_override,
                    bench_name=bench_name
                )

    else:
        raise ValueError(f"Bad question source {args.question_source}.")
