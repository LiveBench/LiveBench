import json
import os
import time
import concurrent.futures
import glob
import shortuuid
import tqdm
from typing import Any

from livebench.agentic_code_runner.sweagent.run_inference import run_agentic_coding_inference
from livebench.common import (
    reorg_answer_file,
)

from livebench.model import ModelConfig, get_api_function


def get_answer(
    question: dict,
    num_choices: int,
    model_config: ModelConfig,
    stream: bool = False,
    answer_file: str | None = None,

):
    """
    Perform inference for a single question.

    Args:
        question: At minimum, a dictionary with a key 'turns' that maps to a list of messages in the conversation, the last of which should ask the question.
        num_choices: The number of model outputs to generate for each question
        answer_file: The path to the file in which to write answers
    """
    
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

            output, num_tokens = get_api_function(model_config.api_provider)(
                model=model_config.api_name,
                messages=messages,
                model_api_kwargs=model_config.api_kwargs,
                api_dict=api_dict,
                stream=stream
            )

            messages.append({"role": "assistant", "content": output})
            turns.append(output)
            total_num_tokens += num_tokens

        choices.append({"index": i, "turns": turns})

    # Dump answers
    ans = {
        "question_id": question["question_id"],
        "answer_id": shortuuid.uuid(),
        "model_id": model_config.display_name.lower(),
        "choices": choices,
        "tstamp": time.time(),
        "total_output_tokens": total_num_tokens,
        "api_info": {
            "provider": model_config.api_provider if model_config.api_provider != 'local' else model_config.api_dict['api_base'],
            "api_name": model_config.api_name,
            "api_kwargs": model_config.api_kwargs
        }
    }

    if answer_file is not None:
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(answer_file, "a") as fout:
            fout.write(json.dumps(ans) + "\n")


def run_inference(
    all_questions: dict[str, dict[str, list[dict[str, Any]]]],
    questions_by_model: dict[str, dict[str, dict[str, set[str]]]],
    model_configs: dict[str, ModelConfig],
    parallel_requests: int | dict[str, int],
    num_choices: int,
    stream: bool,
    bench_name_root: str = "live_bench",
):
    """
    Perform inference on questions for multiple models, organized by category and task.

    Args:
        all_questions: Dictionary mapping category -> task -> list of question objects
        questions_by_model: Dictionary mapping model name -> category -> task -> set of question ids
        model_configs: Dictionary mapping model name -> ModelConfig
        parallel_requests: Number of parallel requests per model (int) or per-model limits (dict)
        num_choices: Number of choices to generate for each question
        stream: Whether to stream model responses
        bench_name_root: Root name for benchmark (defaults to "live_bench")
    """
    
    def process_model_questions(model_name: str, model_config: ModelConfig):
        """Process all questions for a single model."""
        print(f'Model API name: {model_config.api_name}')
        
        model_questions = questions_by_model.get(model_name, {})
        total_questions = sum(
            len(question_ids) 
            for category_tasks in model_questions.values() 
            for question_ids in category_tasks.values()
        )
        
        if total_questions == 0:
            print(f'No questions found for model {model_name}')
            return
            
        print(f'Evaluating {total_questions} questions with model {model_name}')
        
        # Get parallel limit for this model
        if isinstance(parallel_requests, dict):
            model_parallel_limit = parallel_requests.get(model_name, 1)
        else:
            model_parallel_limit = parallel_requests
        
        # Separate agentic coding questions from regular questions
        agentic_coding_questions = []
        agentic_coding_task_info = {}
        regular_question_tasks = []
        
        for category, tasks in model_questions.items():
            for task, question_ids in tasks.items():
                if not question_ids:
                    continue
                    
                # Filter questions by IDs
                category_task_questions = all_questions.get(category, {}).get(task, [])
                filtered_questions = [
                    q for q in category_task_questions 
                    if q.get('question_id') in question_ids
                ]
                
                if not filtered_questions:
                    continue
                
                # Check if this is agentic coding
                if 'agentic_coding' in category:
                    # Collect all agentic coding questions for this model
                    agentic_coding_questions.extend(filtered_questions)
                    # Store task information for each question
                    task_key = f"{category}/{task}"
                    for question in filtered_questions:
                        agentic_coding_task_info[question['question_id']] = {
                            'task_key': task_key,
                            'answer_file': os.path.join(
                                bench_name_root, category, task, "model_answer", 
                                f"{model_config.display_name}.jsonl"
                            ),
                            'bench_name': f"{bench_name_root}/{category}/{task}"
                        }
                else:
                    # Create answer file path for regular questions
                    answer_file = os.path.join(
                        bench_name_root, category, task, "model_answer", 
                        f"{model_config.display_name}.jsonl"
                    )
                    
                    bench_name = f"{bench_name_root}/{category}/{task}"
                    
                    for question in filtered_questions:
                        regular_question_tasks.append((question, answer_file, bench_name))
        
        # Process agentic coding questions if any
        if agentic_coding_questions:
            print(f'Processing {len(agentic_coding_questions)} agentic coding questions for model {model_name}')
            
            run_agentic_coding_inference(
                questions=agentic_coding_questions,
                model_config=model_config,
                num_choices=num_choices,
                task_info=agentic_coding_task_info,
                parallel=model_parallel_limit,
            )
        
        # Process regular questions if any
        if not regular_question_tasks:
            if not agentic_coding_questions:
                print(f'No valid questions found for model {model_name}')
            return
        
        print(f'Processing {len(regular_question_tasks)} regular questions for model {model_name}')
        
        # Process regular questions with parallelization
        if model_parallel_limit == 1:
            for question, answer_file, bench_name in tqdm.tqdm(regular_question_tasks, desc=f"Processing {model_name}"):
                get_answer(
                    question=question,
                    num_choices=num_choices,
                    stream=stream,
                    answer_file=answer_file,
                    model_config=model_config,
                )
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=model_parallel_limit) as executor:
                futures = []
                for question, answer_file, bench_name in regular_question_tasks:
                    future = executor.submit(
                        get_answer,
                        question=question,
                        num_choices=num_choices,
                        stream=stream,
                        answer_file=answer_file,
                        model_config=model_config,
                    )
                    futures.append(future)

                for future in tqdm.tqdm(
                    concurrent.futures.as_completed(futures), 
                    total=len(futures),
                    desc=f"Processing {model_name}"
                ):
                    future.result()
        
        # Reorganize answer files for regular questions
        processed_files = set()
        for _, answer_file, _ in regular_question_tasks:
            if answer_file not in processed_files:
                if os.path.exists(answer_file):
                    reorg_answer_file(answer_file)
                processed_files.add(answer_file)
    
    # Process all models in parallel
    models_to_process = [
        (model_name, model_config) 
        for model_name, model_config in model_configs.items() 
        if model_name in questions_by_model
    ]
    
    if not models_to_process:
        print("No models found with questions to process")
        return
    
    if len(models_to_process) == 1:
        # Single model - no need for additional threading
        model_name, model_config = models_to_process[0]
        process_model_questions(model_name, model_config)
    else:
        # Multiple models - process in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(models_to_process)) as executor:
            model_futures = []
            for model_name, model_config in models_to_process:
                future = executor.submit(process_model_questions, model_name, model_config)
                model_futures.append(future)
            
            # Wait for all models to complete
            for future in concurrent.futures.as_completed(model_futures):
                future.result()