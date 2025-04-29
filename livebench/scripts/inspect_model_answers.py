#!/usr/bin/env python3
"""
Compare answers from one or two models for a specific question or all questions in a benchmark.

Usage:
python compare_model_answers.py --model1 <model1_id> [--model2 <model2_id>] --question_id <question_id>
or
python compare_model_answers.py --model1 <model1_id> [--model2 <model2_id>] --bench-name <bench_name>
"""

import argparse
import json
import sys
import glob
import os
import io
import re
import contextlib
from typing import Any
from livebench.model import get_model_config
from livebench.common import load_questions_jsonl
from livebench.gen_ground_truth_judgment import play_a_match_gt, MatchSingle
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markup import escape
from rich import box

MISTAKE_EXPLANATION_PROMPT = """You are an expert in large language model benchmarking.
Your task is to explain why a model's answer is incorrect.

You will be given a question, the model's answer, and the ground truth answer.
You may also be given some debug output from the evaluation of the model's answer.

You will need to explain why the model's answer is incorrect.
Cite specific details from the question, model's answer, ground truth answer, and debug output to support your explanation.
Try to describe the more general type of mistake rather than pointing out all specific details.
For instance, for a coding question, you may note how a model misunderstood the question, or how it failed to correctly use a library.

Keep your explanation very concise; it should be a single paragraph that gets right to the point in highlighting the mistake.
ONLY discuss mistakes that you think directly led to the model receiving a score less than 1.
For instance, in a coding task, you may notice other issues in the model response, but you should only discuss mistakes that seem to have led to test case failures.

Your response should be 1-2 sentences at most.

Question:
{question}

Ground Truth Answer:
{ground_truth_answer}

Model's Answer:
{model_answer}
"""

MISTAKE_CODING_QUESTION_ADDENDUM = """
The question is a coding question. You will also be provided with the test suite used to evaluate the model's answer.
The ground truth answer has been confirmed to pass all the tests in the test suite.
Based on the test suite, describe the specific reasons why the model's answer failed to pass the tests.
Note specifically whether the issue actually seems to be with the test suite, for instance if a function call is not mocked correctly.

You should be very careful to only discuss mistakes that seem to have directly led to the model receiving a score less than 1, which occurs if any test case fails.

Note: the model should be allowed to use any libraries that are imported in the question, and should be allowed to use any library function in those libraries that are relevant to the question.
So if the model uses a library function that serves the correct purpose, but the test suite fails because the function call is not mocked correctly, this is a mistake in the test suite, not the model's answer, and you should note that.

Classify the mistake as one of the following:
 - MODEL MISTAKE: the model actually made a mistake, perhaps by improperly using a library function, or by not adhering to the specifications in the QUESTION
 - INCOMPLETE SPECIFICATION: the model's answer adheres to the specifications in the QUESTION, but the test suite implicitly has additional requirements that are not specified in the QUESTION
 - MISTAKE IN THE TEST SUITE: the test suite has an issue, such as a function call that is not mocked correctly

Note: if your explanation is of the form "the model does _something_ but the test suite expects _something else_", this is likely a MISTAKE IN THE TEST SUITE or INCOMPLETE SPECIFICATION, not a MODEL MISTAKE.
It's only if the question specification prevents what the model did from being correct that it should be considered a MODEL MISTAKE.
It is a MISTAKE IN THE TEST SUITE if the thing that the model did is, e.g. using a function call that wasn't mocked but still serves the correct purpose, or the test suite asserts the use of only one of many possible library functions to do a task.
It is an INCOMPLETE SPECIFICATION if the test suite implicitly has additional requirements that are not specified in the QUESTION, such as a particular format or a magic string that the model should have used.

Test Suite:
{test_suite}

Your response should be of the format:
Mistake Type: <mistake_type>
Mistake Explanation: <explanation>
"""

def load_cached_inspection(file_path: str) -> dict[str, dict[str, str]]:
    """Load inspection data from an existing markdown file.
    
    Args:
        file_path: Path to the inspection file
        
    Returns:
        Dictionary mapping question_ids to a dictionary containing model_answer, debug_output, and mistake_explanation
    """
    if not os.path.exists(file_path):
        return {}
    
    cached_data = {}
    current_question_id = None
    current_section = None
    current_content = []
    
    with open(file_path, 'r') as f:
        for line in f:
            # Check for question ID
            question_id_match = re.match(r'# Question ID: (.+)', line)
            if question_id_match:
                # Save previous question data if it exists
                if current_question_id and current_section:
                    if current_question_id not in cached_data:
                        cached_data[current_question_id] = {}
                    cached_data[current_question_id][current_section] = '\n'.join(current_content).strip()
                
                # Start new question
                current_question_id = question_id_match.group(1)
                current_section = None
                current_content = []
                continue
            
            # Check for section headers
            if line.startswith('## Model Answer'):
                if current_question_id and current_section:
                    if current_question_id not in cached_data:
                        cached_data[current_question_id] = {}
                    cached_data[current_question_id][current_section] = '\n'.join(current_content).strip()
                current_section = 'model_answer'
                current_content = []
                continue
            elif line.startswith('## Debug Output'):
                if current_question_id and current_section:
                    if current_question_id not in cached_data:
                        cached_data[current_question_id] = {}
                    cached_data[current_question_id][current_section] = '\n'.join(current_content).strip()
                current_section = 'debug_output'
                current_content = []
                continue
            elif line.startswith('## Mistake Explanation'):
                if current_question_id and current_section:
                    if current_question_id not in cached_data:
                        cached_data[current_question_id] = {}
                    cached_data[current_question_id][current_section] = '\n'.join(current_content).strip()
                current_section = 'mistake_explanation'
                current_content = []
                continue
            elif line.startswith('---'):
                # End of a question
                if current_question_id and current_section:
                    if current_question_id not in cached_data:
                        cached_data[current_question_id] = {}
                    cached_data[current_question_id][current_section] = '\n'.join(current_content).strip()
                current_section = None
                current_content = []
                continue
            
            # Process code block markers for debug output
            if current_section == 'debug_output':
                if line.strip() == '```':
                    continue
            
            # Append content to current section
            if current_section:
                current_content.append(line.rstrip())
    
    # Save last question data if needed
    if current_question_id and current_section:
        if current_question_id not in cached_data:
            cached_data[current_question_id] = {}
        cached_data[current_question_id][current_section] = '\n'.join(current_content).strip()
    
    return cached_data

def write_all_questions_to_file(file_path: str, collected_data: dict[str, dict]):
    """Write all collected question data to a file.
    
    Args:
        file_path: File path to write to
        collected_data: Dictionary mapping question_ids to data dictionaries
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Load existing data to merge with new data
    existing_data = {}
    if os.path.exists(file_path):
        existing_data = load_cached_inspection(file_path)
    
    # Merge existing data with new data
    merged_data = {**existing_data, **collected_data}
    
    # Write all data to file
    with open(file_path, 'w') as f:
        for question_id, data in merged_data.items():
            f.write(f"# Question ID: {question_id}\n")
            
            if 'model_answer' in data:
                f.write("## Model Answer\n")
                f.write(f"{data['model_answer']}\n\n")
            
            if 'debug_output' in data and data['debug_output']:
                f.write("## Debug Output\n")
                f.write(f"```\n{data['debug_output']}\n```\n\n")
            
            if 'mistake_explanation' in data and data['mistake_explanation']:
                f.write("## Mistake Explanation\n")
                f.write(f"{data['mistake_explanation']}\n\n")
            
            f.write("---\n\n")

def get_mistake_explanation(question_text: str, ground_truth_answer: str, model_answer: str, test_suite: str | None = None, debug_output: str | None = None) -> str:
    """Capture debug output from play_a_match_gt function.
    
    Args:
        question_text: Text of the question
        ground_truth_answer: Ground truth answer
        model_answer: Model's answer
        debug_output: Debug output (optional)
        
    Returns:
        Mistake explanation as string
    """
    prompt = MISTAKE_EXPLANATION_PROMPT.format(question=question_text, ground_truth_answer=ground_truth_answer, model_answer=model_answer)
    if debug_output:
        prompt += f"\n\nDebug Output:\n{debug_output}"

    if test_suite:
        prompt += MISTAKE_CODING_QUESTION_ADDENDUM.format(test_suite=test_suite)

    prompt += "\n\nExplanation:"
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""

def load_model_answers(model_id: str) -> dict[str, Any]:
    """Load all of a model's answers.
    
    Args:
        model_id: ID of the model whose answers to load
        
    Returns:
        Dictionary mapping tasks to question_ids to answer objects
    """
    model_files = glob.glob(f"**/{model_id}.jsonl", recursive=True)
    
    if not model_files:
        print(f"Error: Could not find answer file for model {model_id}.")
        sys.exit(1)

    answers = {}

    for filename in model_files:
        task = filename.split("data/")[1].split("/model_answer")[0]
        answers[task] = {}
        with open(filename, 'r') as f:
            for line in f:
                try:
                    answer_obj = json.loads(line.strip())
                    if "question_id" in answer_obj:
                        answers[task][str(answer_obj["question_id"])] = answer_obj
                except json.JSONDecodeError:
                    continue
    
    return answers


def load_model_scores(task: str) -> dict[str, dict[str, float]]:
    """Load scores for models from the judgment file.
    
    Args:
        task: The task to load scores for
        
    Returns:
        Dictionary mapping question_ids to model_ids to scores
    """
    judgment_path = f"data/{task}/model_judgment/ground_truth_judgment.jsonl"
    
    if not os.path.exists(judgment_path):
        return {}
    
    scores = {}
    
    with open(judgment_path, 'r') as f:
        for line in f:
            try:
                judgment_obj = json.loads(line.strip())
                question_id = str(judgment_obj.get("question_id"))
                model_id = judgment_obj.get("model")
                score = judgment_obj.get("score")
                
                if question_id and model_id and score is not None:
                    if question_id not in scores:
                        scores[question_id] = {}
                    scores[question_id][model_id] = score
            except json.JSONDecodeError:
                continue
    
    return scores


def get_answer_for_question(answers: dict[str, dict[str, Any]], question_id: str) -> tuple[str, dict[str, Any]] | None:
    """Get the answer for a specific question from the loaded answers.
    
    Args:
        answers: Dictionary of answers indexed by task and question_id
        question_id: The ID of the question to get the answer for
        
    Returns:
        Tuple of (task, answer_object) for the specified question, or None if not found
    """
    question_id = str(question_id)
    for task, task_answers in answers.items():
        if question_id in task_answers:
            return task, task_answers[question_id]
    return None


def load_question_content(task: str, question_id: str) -> dict[str, Any] | None:
    """Load the question content from question.jsonl in the parent directory.
    
    Args:
        task: The task to load the question content for
        question_id: The ID of the question to load
        
    Returns:
        Question object for the specified question, or None if not found
    """
    question_path = f"data/{task}/question.jsonl"
    
    if not os.path.exists(question_path):
        return None
    
    questions = load_questions_jsonl(question_path, question_ids=[question_id])
    
    if not questions:
        return None
        
    return questions[0]


def get_debug_output(question: dict[str, Any], model_id: str, answer: dict[str, Any]) -> str:
    """Capture debug output from play_a_match_gt function.
    
    Args:
        question: The question object
        model_id: The model ID
        answer: The answer object
        
    Returns:
        Debug output as string
    """
    # Create a match object for the question and answer
    match = MatchSingle(
        question=question,
        model=model_id,
        answer=answer
    )
    
    # Capture stdout to get debug output
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        try:
            play_a_match_gt(match, output_file=None, debug=True)
        except Exception as e:
            return f"Error evaluating answer: {str(e)}"
    
    return f.getvalue()

def process_single_question(question_id, model_id, model_answers, include_judgment_debug, console, 
                          include_mistake_explanations=False, output_to_file=False):
    """Process a single question for a specific model and return the data needed for comparison.
    
    Args:
        question_id: ID of the question to process
        model_id: ID of the model to process
        model_answers: Dictionary of answers for the model
        include_judgment_debug: Whether to include judgment debug output
        console: Rich console for output
        include_mistake_explanations: Whether to include mistake explanations
        output_to_file: Whether to output results to file
        
    Returns:
        Tuple of (task, question_content, model_answer, model_answer_obj, model_score, model_debug, model_mistake_explanation)
        or None if the model doesn't have an answer for the question
    """
    # Get answer for the specified question
    task_and_model_answer = get_answer_for_question(model_answers, question_id)

    if task_and_model_answer is None:
        console.print(f"No answer found for question {question_id} from model {model_id}", style="bold red")
        return None
    
    task, model_answer_obj = task_and_model_answer
    model_answer: str = model_answer_obj['choices'][0]['turns'][0]

    if '</think>' in model_answer:
        model_answer = model_answer.split('</think>')[1].strip()
    elif '<think>' in model_answer:
        # if there's a <think> without a </think>, then we hit max tokens while thinking
        # so there's no actual answer
        model_answer = ''


    
    # Load score for the task
    model_scores = load_model_scores(task)
    model_score = model_scores.get(question_id, {}).get(model_id)
    
    # Check for cached data
    model_cached_data = {}
    
    model_file_path = f"answer_inspections/{model_id}-{task.replace('/', '_')}-inspection.md"
    if os.path.exists(model_file_path):
        cached_data = load_cached_inspection(model_file_path)
        if question_id in cached_data:
            model_cached_data = cached_data[question_id]
    
    # Find the question content
    question_content = load_question_content(task, question_id)

    if 'partial_solution' in question_content:
        if '```python' in model_answer:
            model_answer = model_answer.split('```python')[1].strip()
        if model_answer.startswith('```'):
            model_answer = model_answer[3:].strip()
        if model_answer.endswith('```'):
            model_answer = model_answer[:-3].strip()
        if not model_answer.startswith(question_content['partial_solution']):
            model_answer = question_content['partial_solution'] + '\n' + model_answer
    
    # Get debug output if include_judgment_debug is set and not in cache
    model_debug = None
    
    # Normalize whitespace in both strings for comparison
    def normalize_whitespace(text):
        # Remove leading/trailing whitespace from each line and normalize spaces
        lines = [line.strip() for line in text.strip().splitlines()]
        # Remove empty lines
        lines = [line for line in lines if line]
        return '\n'.join(lines)
    
    use_model_cache = 'model_answer' in model_cached_data and (
        model_cached_data['model_answer'].strip() == model_answer.strip() or
        normalize_whitespace(model_cached_data['model_answer']) == normalize_whitespace(model_answer)
    )
    
    
    if use_model_cache:
        console.print(escape(f"Using cached judgment debug for model {model_id} for question {question_id}"), style="green")
        if 'debug_output' in model_cached_data:
            model_debug = model_cached_data['debug_output']
    else:
        if include_judgment_debug and question_content:
            model_debug = get_debug_output(question_content, model_id, model_answer_obj)
    
    # Generate mistake explanation if needed and not in cache
    model_mistake_explanation = None
    mistake_type = None
    if question_content and "ground_truth" in question_content:
        if model_score is not None and model_score != 1:
            if use_model_cache and 'mistake_explanation' in model_cached_data:
                console.print(escape(f"Using cached mistake explanation for model {model_id} for question {question_id}"), style="green")
                model_mistake_explanation = model_cached_data['mistake_explanation']
            elif include_mistake_explanations:
                question_text = str(question_content['turns'][0]) if 'turns' in question_content and len(question_content['turns']) > 0 else ""
                ground_truth = str(question_content.get("ground_truth", ""))
                if question_content['task'] in ['BCB_generation', 'BCB_completion']:
                    ground_truth = question_content['code_prompt'] + '\n' + ground_truth
                if not model_debug:
                    model_debug = get_debug_output(question_content, model_id, model_answer_obj)
                if 'stat timeout' in model_debug:
                    model_mistake_explanation = "TIMEOUT"
                    mistake_type = "MODEL MISTAKE"
                else:
                    model_mistake_explanation = get_mistake_explanation(
                        question_text=question_text,
                        ground_truth_answer=ground_truth,
                        model_answer=str(model_answer),
                        test_suite=question_content['tests'] if 'tests' in question_content else None,
                        debug_output=model_debug
                    )
                    if 'Mistake Type' in model_mistake_explanation:
                        mistake_type = model_mistake_explanation.split('Mistake Type: ')[1].split('\n')[0].strip()
                    
    
    # Collect data for writing to file
    model_collected_data = {}
    
    # If output_to_file is set, collect data for writing
    if output_to_file:
        # Only collect data if we didn't use cache and when score is not 1
        if model_answer and model_score is not None and model_score != 1 and not use_model_cache:
            model_collected_data[question_id] = {
                'model_answer': model_answer,
                'debug_output': model_debug,
                'mistake_explanation': model_mistake_explanation
            }
        
        # Write collected data to file
        if model_collected_data:
            file_path = f"answer_inspections/{model_id}-{task.replace('/', '_')}-inspection.md"
            write_all_questions_to_file(file_path, model_collected_data)
    
    return task, question_content, model_answer, model_answer_obj, model_score, model_debug, model_mistake_explanation, mistake_type


def process_model_comparison(question_id, model1, model2, model1_answers, model2_answers, include_judgment_debug, 
                            console, include_mistake_explanations=False, output_to_file=False):
    """Process and compare model answers for a specific question.
    
    Args:
        question_id: ID of the question to process
        model1: First model ID
        model2: Second model ID (can be None)
        model1_answers: Dictionary of answers for model1
        model2_answers: Dictionary of answers for model2 (can be None)
        include_judgment_debug: Whether to include judgment debug output
        console: Rich console for output
        include_mistake_explanations: Whether to include mistake explanations
        output_to_file: Whether to output results to file
    """
    # Process model1
    model1_result = process_single_question(question_id, model1, model1_answers, include_judgment_debug, 
                                         console, include_mistake_explanations, output_to_file)
    
    if model1_result is None:
        return
        
    task, question_content, model1_answer, model1_answer_obj, model1_score, model1_debug, model1_mistake_explanation, model1_mistake_type = model1_result
    
    # Display question content once
    if question_content and 'turns' in question_content and len(question_content['turns']) > 0:
        console.print(Panel(str(question_content['turns'][0]), title="Question Content", expand=True))
        console.print("\n")
    
    # Process model2 if it exists
    model2_result = None
    model2_answer = None
    model2_answer_obj = None
    model2_score = None
    model2_debug = None
    model2_mistake_explanation = None
    model2_mistake_type = None

    if model2 and model2_answers:
        model2_result = process_single_question(question_id, model2, model2_answers, 
                                             include_judgment_debug, console, 
                                             include_mistake_explanations, output_to_file)
        if model2_result:
            _, _, model2_answer, model2_answer_obj, model2_score, model2_debug, model2_mistake_explanation, model2_mistake_type = model2_result
    
    # Display the comparison
    display_model_comparison(model1, model2, model1_answer, model2_answer, model1_score, model2_score, 
                           question_content, include_judgment_debug, 
                           console, include_mistake_explanations, model1_debug, model2_debug, 
                           model1_mistake_explanation, model2_mistake_explanation)

    return model1_mistake_type, model2_mistake_type


def process_benchmark(bench_name, model1, model2, model1_answers, model2_answers, include_judgment_debug, 
                     console, only_incorrect=False, include_mistake_explanations=False, output_to_file=False,
                     exclude_ids=None):
    if bench_name not in model1_answers:
        console.print(f"No answers found for benchmark {bench_name} from model {model1}", style="bold red")
        return
    
    if model2 and model2_answers and bench_name not in model2_answers:
        console.print(f"No answers found for benchmark {bench_name} from model {model2}", style="bold red")
        model2 = None
        model2_answers = None
    
    # Load all questions for this benchmark
    question_path = f"data/{bench_name}/question.jsonl"
    if not os.path.exists(question_path):
        console.print(f"Question file not found for benchmark {bench_name}", style="bold red")
        return
    
    questions = load_questions_jsonl(question_path)
    model_scores = load_model_scores(bench_name)

    model1_mistake_types = {}
    model2_mistake_types = {}

    
    for question in questions:
        question_id = str(question["question_id"])
        
        # Skip if question_id is in exclude_ids
        if exclude_ids and question_id in exclude_ids:
            console.print(f"Skipping question {question_id} - in exclude list", style="yellow")
            continue
        
        # Skip if the answer is not found from model1
        if question_id not in model1_answers[bench_name]:
            console.print(f"Skipping question {question_id} - answer not found from model {model1}", style="yellow")
            continue
        
        # Handle model2 if it exists
        if model2 and model2_answers and question_id not in model2_answers[bench_name]:
            console.print(f"Skipping question {question_id} - answer not found from model {model2}", style="yellow")
            continue
        
        # Skip if only_incorrect is set and both models got a perfect score (or missing scores)
        if only_incorrect:
            model1_score = model_scores.get(question_id, {}).get(model1)
            model2_score = None
            if model2:
                model2_score = model_scores.get(question_id, {}).get(model2)
            
            # Only process questions where at least one model got a score other than 1
            if (model1_score == 1 or model1_score is None) and (model2_score == 1 or model2_score is None):
                continue
        
        # Print separator for better readability
        console.print(f"\n\n{'='*80}", style="bold")
        
        # Print question ID
        console.print(f"Question ID: {question_id}\n", style="bold")
        
        # Process and compare model answers
        model1_mistake_type, model2_mistake_type = process_model_comparison(question_id, model1, model2, model1_answers, model2_answers, include_judgment_debug,
                                console, include_mistake_explanations, output_to_file)
        if model1_mistake_type:
            model1_mistake_types[model1_mistake_type] = model1_mistake_types.get(model1_mistake_type, []) + [question_id]
        if model2_mistake_type:
            model2_mistake_types[model2_mistake_type] = model2_mistake_types.get(model2_mistake_type, []) + [question_id]
    
    if model1_mistake_types != {}:
        print('Model 1 mistake types:')
        for mistake_type, question_ids in model1_mistake_types.items():
            print(f"{mistake_type}: {question_ids}")
    if model2 and model2_mistake_types != {}:
        print('Model 2 mistake types:')
        for mistake_type, question_ids in model2_mistake_types.items():
            print(f"{mistake_type}: {question_ids}")


def main():
    parser = argparse.ArgumentParser(description="Compare answers from one or two models for a specific question or all questions in a benchmark.")
    parser.add_argument("--bench-name", help="Name of the benchmark to compare all questions for")
    parser.add_argument("--question-id", help="ID of the question to compare answers for")
    parser.add_argument("--model1", required=True, help="First model ID or path to JSONL file")
    parser.add_argument("--model2", help="Second model ID or path to JSONL file (optional)")
    parser.add_argument("--include-judgment-debug", action="store_true", help="Include debug output from judgment")
    parser.add_argument("--only-incorrect", action="store_true", help="When processing a benchmark, only show questions where at least one model received a score other than 1")
    parser.add_argument("--include-mistake-explanations", action="store_true", help="Include explanations for why a model's answer is incorrect when score is not 1")
    parser.add_argument("--output-to-file", action="store_true", help="Output results to markdown files in answer_inspections directory")
    parser.add_argument("--exclude-ids", nargs="+", help="List of question IDs to exclude when processing a benchmark")
    args = parser.parse_args()

    if not args.bench_name and not args.question_id:
        parser.error("Either --bench-name or --question_id must be specified")

    model1 = get_model_config(args.model1).display_name
    
    # Load answers for model1
    model1_answers = load_model_answers(model1)
    
    # Only load model2 answers if model2 is provided
    model2 = None
    model2_answers = None
    if args.model2:
        model2 = get_model_config(args.model2).display_name
        model2_answers = load_model_answers(model2)
    
    # Create rich console
    console = Console()

    if args.question_id:
        # Process a single question for each model
        console.print(f"Question ID: {args.question_id}\n", style="bold")
        
        # Process and compare model answers
        process_model_comparison(args.question_id, model1, model2, model1_answers, model2_answers, 
                                args.include_judgment_debug, console, args.include_mistake_explanations, 
                                args.output_to_file)
    elif args.bench_name:
        # Process all questions in the benchmark
        process_benchmark(args.bench_name, model1, model2, model1_answers, model2_answers, 
                         args.include_judgment_debug, console, args.only_incorrect, 
                         args.include_mistake_explanations, args.output_to_file, 
                         args.exclude_ids)


def display_model_comparison(model1, model2, model1_answer, model2_answer, model1_score, model2_score,
                           question_content, include_judgment_debug, 
                           console, include_mistake_explanations=False, model1_debug=None, model2_debug=None,
                           model1_mistake_explanation=None, model2_mistake_explanation=None):
    # Check if ground truth exists
    has_ground_truth = question_content and "ground_truth" in question_content
    ground_truth = question_content.get("ground_truth", None) if has_ground_truth else None

    if question_content['task'] in ['BCB_generation', 'BCB_completion'] and not ground_truth.startswith(question_content['code_prompt']):
        ground_truth = question_content['code_prompt'] + '\n' + ground_truth
    
    # Escape rich markup in text to prevent formatting interpretation
    if ground_truth:
        ground_truth = escape(str(ground_truth))
    if model1_answer:
        model1_answer = escape(str(model1_answer))
    if model2_answer:
        model2_answer = escape(str(model2_answer))
    
    # Escape rich markup in debug output
    if model1_debug:
        model1_debug = escape(str(model1_debug))
    if model2_debug:
        model2_debug = escape(str(model2_debug))
    
    # Escape rich markup in explanations
    if model1_mistake_explanation:
        model1_mistake_explanation = escape(str(model1_mistake_explanation))
    if model2_mistake_explanation:
        model2_mistake_explanation = escape(str(model2_mistake_explanation))
    
    if model2:
        # Comparative display with two models
        table = Table(show_header=True, expand=True, box=box.SQUARE)
        
        # Add ground truth column if it exists
        if has_ground_truth:
            table.add_column("Ground Truth", ratio=1, no_wrap=False)
        
        # Add model name and score if available
        model1_header = model1
        if model1_score is not None:
            model1_header += f" (Score: {model1_score})"
        
        model2_header = model2
        if model2_score is not None:
            model2_header += f" (Score: {model2_score})"
        
        table.add_column(model1_header, ratio=1, no_wrap=False)
        table.add_column(model2_header, ratio=1, no_wrap=False)

        # Add the row with answers
        if has_ground_truth:
            table.add_row(ground_truth, model1_answer, model2_answer)
        else:
            table.add_row(model1_answer, model2_answer)
        
        # If include_judgment_debug is set, add debug output to the table
        if include_judgment_debug and (model1_debug or model2_debug):
            # Add a horizontal border between rows
            table.add_section()
            
            # Add the row with debug output
            if has_ground_truth:
                table.add_row("", model1_debug, model2_debug)
            else:
                table.add_row(model1_debug, model2_debug)
        
        # If include_mistake_explanations is set and we have explanations, add them to the table
        if include_mistake_explanations and (model1_mistake_explanation or model2_mistake_explanation):
            # Add a horizontal border between rows
            table.add_section()
            
            # Add the row with mistake explanations
            if has_ground_truth:
                table.add_row("", model1_mistake_explanation or "", model2_mistake_explanation or "")
            else:
                table.add_row(model1_mistake_explanation or "", model2_mistake_explanation or "")
    else:
        # Display for single model
        table = Table(show_header=True, expand=True, box=box.SQUARE)
        
        # Add ground truth column if it exists
        if has_ground_truth:
            table.add_column("Ground Truth", ratio=1, no_wrap=False)
        
        # Add model name and score if available
        model1_header = model1
        if model1_score is not None:
            model1_header += f" (Score: {model1_score})"
        
        table.add_column(model1_header, ratio=1, no_wrap=False)

        # Add the row with model1 answer
        if has_ground_truth:
            table.add_row(ground_truth, model1_answer)
        else:
            table.add_row(model1_answer)
        
        # If include_judgment_debug is set, add debug output
        if include_judgment_debug and model1_debug:
            # Add a horizontal border between rows
            table.add_section()
            
            # Add the row with debug output
            if has_ground_truth:
                table.add_row("", model1_debug)
            else:
                table.add_row(model1_debug)
        
        # If include_mistake_explanations is set and we have an explanation, add it to the table
        if include_mistake_explanations and model1_mistake_explanation:
            # Add a horizontal border between rows
            table.add_section()
            
            # Add the row with mistake explanation
            if has_ground_truth:
                table.add_row("", model1_mistake_explanation)
            else:
                table.add_row(model1_mistake_explanation)
    
    # Print the table
    console.print(table)


if __name__ == "__main__":
    main()