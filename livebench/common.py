"""
Common data structures and utilities.
"""

import ast
import dataclasses
from datasets import load_dataset, Dataset
from datetime import datetime
import glob
import json
import os

import re
import time
from typing import Optional
import openai
import anthropic

# API setting constants
API_MAX_RETRY = 16
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"

# Extract scores from judgments
two_score_pattern = re.compile("\[\[(\d+\.?\d*),\s?(\d+\.?\d*)\]\]")
two_score_pattern_backup = re.compile("\[(\d+\.?\d*),\s?(\d+\.?\d*)\]")
one_score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
one_score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")

# Huggingface and dataset-related constants
LIVE_BENCH_HF_ORGANIZATION = "livebench"
LIVE_BENCH_DATA_SUPER_PATH = "live_bench"
LIVE_BENCH_CATEGORIES = [
    'coding',
    'data_analysis',
    'instruction_following',
    'math',
    'reasoning',
    'language',
]

@dataclasses.dataclass
class MatchSingle:
    question: dict
    model: str
    answer: dict
    ref_answer: dict = None
    multi_turn: bool = False


def get_categories_tasks(bench_name: str):
    split_bench_name = bench_name.rstrip('/').split('/')
    assert(split_bench_name[0] == 'live_bench')
    if len(split_bench_name) == 1:
        # specify entire bench

        categories = {
            category_name : get_hf_dataset(category_name)
            for category_name in LIVE_BENCH_CATEGORIES
        }

        tasks = {
            category_name : get_tasks_from_hf_category(categories[category_name])
            for category_name in LIVE_BENCH_CATEGORIES
        }

    else:
        # specify a category or task
        category_name = split_bench_name[1]

        categories = {
            category_name : get_hf_dataset(category_name)
        }

        if len(split_bench_name) == 2:
            tasks = {
                category_name : get_tasks_from_hf_category(categories[category_name])
            } 
        else:
            assert(len(split_bench_name) == 3)
            task_name = split_bench_name[2]
            tasks = {
                category_name : [
                    task_name
                ]
            } 

    return categories, tasks


def get_hf_dataset(dataset_name: str, split='test'):
    return load_dataset(f"{LIVE_BENCH_HF_ORGANIZATION}/{dataset_name}", split=split)


def get_tasks_from_hf_category(category: Dataset):
    return list(set(category["task"]))


def load_answers_judgments():
    model_judgment_dataset = get_hf_dataset("model_judgment", split="leaderboard")
    model_answer_dataset   = get_hf_dataset("model_answer", split="leaderboard")

    model_judgment = {
        category_name : [example for example in model_judgment_dataset.filter(lambda row: row["category"] == category_name)]
        for category_name in LIVE_BENCH_CATEGORIES
    }

    model_answer = {
        category_name : [example for example in model_answer_dataset.filter(lambda row: row["category"] == category_name)]
        for category_name in LIVE_BENCH_CATEGORIES
    }

    return model_answer, model_judgment


def load_questions(category: Dataset, livebench_releases: set, task_name: Optional[str], begin: Optional[int], end: Optional[int]):
    """Load questions from a file."""
    if task_name is not None:
        questions = [example for example in category.filter(lambda row: row["task"] == task_name)]
    else:
        questions = list(category)
    questions = questions[begin:end]
    for q in questions:
        if 'livebench_release_date' in q.keys() and isinstance(q['livebench_release_date'], datetime):
            q['livebench_release_date'] = datetime.strftime(q['livebench_release_date'], '%Y-%m-%d')
        if 'release_date' in q.keys() and isinstance(q['release_date'], datetime):
            q['release_date'] = datetime.strftime(q['release_date'], '%Y-%m-%d')
        if 'original_json' in q.keys() and 'contest_date' in q['original_json'].keys() and isinstance(q['original_json']['contest_date'], datetime):
            q['original_json']['contest_date'] = datetime.strftime(q['original_json']['contest_date'], '%Y-%m-%d %H:%M:%S')
    questions = [q for q in questions if q['livebench_release_date'] in livebench_releases]
    return questions


def load_questions_jsonl(question_file: str, livebench_releases: set, begin: Optional[int], end: Optional[int]):
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    questions = questions[begin:end]
    questions = [q for q in questions if q['livebench_release_date'] in livebench_releases]
    return questions


def load_model_answers(answer_dir: str):
    """Load model answers.

    The return value is a python dict of type:
    Dict[model_name: str -> Dict[question_id: int -> answer: dict]]
    """
    filenames = glob.glob(os.path.join(answer_dir, "*.jsonl"))
    filenames.sort()
    model_answers = {}

    for filename in filenames:
        model_name = os.path.basename(filename)[:-len('.jsonl')]
        answer = {}
        with open(filename) as fin:
            for line in fin:
                line = json.loads(line)
                answer[line["question_id"]] = line
        model_answers[model_name] = answer

    return model_answers


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


def make_match_single(
    questions,
    models,
    model_answers,
    multi_turn=False,
):
    matches = []
    for q in questions:
        if multi_turn and len(q["turns"]) != 2:
            continue
        for i in range(len(models)):
            q_id = q["question_id"]
            m = models[i]
            a = model_answers[m][q_id]

            matches.append(
                MatchSingle(
                    dict(q), m, a, multi_turn=multi_turn
                )
            )
    return matches

def chat_completion_openai(model, conv, temperature, max_tokens, api_dict=None):
    if api_dict is not None:
        openai.api_base = api_dict["api_base"]
        openai.api_key = api_dict["api_key"]
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            messages = conv.to_openai_api_messages()
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                n=1,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = response["choices"][0]["message"]["content"]
            break
        except Exception as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)

    return output

def chat_completion_deepseek(model, conv, temperature, max_tokens, api_dict=None):
    if api_dict is not None and "api_key" in api_dict:
        api_key = api_dict["api_key"]
    else:
        api_key = os.environ["DEEPSEEK_API_KEY"]
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            print('sleeping for 3 sec')
            time.sleep(3)
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
            messages = conv.to_openai_api_messages()
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=1,
                stream=False
            )
            output = response.choices[0].message.content
            break
        except Exception as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)

    return output

def chat_completion_nvidia(model, conv, temperature, max_tokens, api_dict=None):
    if api_dict is not None and "api_key" in api_dict:
        api_key = api_dict["api_key"]
    else:
        api_key = os.environ["NVIDIA_API_KEY"]
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            print('sleeping for 2 sec')
            time.sleep(2)
            if "nvidia/" in model:
                full_model_name = model
            else:
                full_model_name = "nvidia/" + model

            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url="https://integrate.api.nvidia.com/v1")
            messages = conv.to_openai_api_messages()
            response = client.chat.completions.create(
                model=full_model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=1,
                stream=False
            )
            output = response.choices[0].message.content
            break
        except Exception as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)

    return output

def chat_completion_vertex(model, conv, temperature, max_tokens, api_dict=None, project_name="DEFAULT"):
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            import vertexai
            from vertexai.preview.generative_models import GenerativeModel, Image
            print('sleeping for 5 sec')
            time.sleep(5)
            vertexai.init(project=project_name, location="us-central1")
            generative_multimodal_model = GenerativeModel(model)
            prompt = conv.messages[0][1]
            prompt = [text for role, text in conv.messages if role == "user"][0]
            response = generative_multimodal_model.generate_content([prompt])
            output = response.candidates[0].content.parts[0].text
            break
        except Exception as e:
            print(e)
            print('sleeping for 5 sec')
            time.sleep(5)

    return output.strip()

def chat_completion_google_generativeai(model, conv, temperature, max_tokens, api_dict=None):
    import google.generativeai as genai
    if api_dict is not None and "api_key" in api_dict:
        api_key = api_dict["api_key"]
    else:
        api_key = os.environ["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE"
        },
    ]
    generation_config = {
        "temperature": temperature,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": max_tokens,
    }

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            print('sleeping for 3 sec')
            time.sleep(3)
            gemini = genai.GenerativeModel(
                model_name=model,
                generation_config=generation_config,
                safety_settings=safety_settings)

            convo = gemini.start_chat(history=[])
            prompt = conv.messages[0][1]
            prompt = [text for role, text in conv.messages if role == "user"][0]

            convo.send_message(prompt)
            output = convo.last.text
            break
        except genai.types.generation_types.StopCandidateException as e:
            print(type(e), e)
            break
        except Exception as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)

    return output

def chat_completion_together(model, conv, temperature, max_tokens, api_dict=None):
    from together import Together
    if api_dict is not None and "api_key" in api_dict:
        api_key = api_dict["api_key"]
    else:
        api_key = os.environ["TOGETHER_API_KEY"]
    client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            prompt = [text for role, text in conv.messages if role == "user"][0]
            stream = client.chat.completions.create(
                model="meta-llama/"+model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            output = ""
            for chunk in stream:
                output += chunk.choices[0].delta.content or ""

        except Exception as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)

    return output

def chat_completion_openai_azure(model, conv, temperature, max_tokens, api_dict=None):
    openai.api_type = "azure"
    openai.api_version = "2023-07-01-preview"
    if api_dict is not None:
        openai.api_base = api_dict["api_base"]
        openai.api_key = api_dict["api_key"]
    else:
        openai.api_base = os.environ["AZURE_OPENAI_ENDPOINT"]
        openai.api_key = os.environ["AZURE_OPENAI_KEY"]

    if "azure-" in model:
        model = model[6:]

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            messages = conv.to_openai_api_messages()
            response = openai.ChatCompletion.create(
                engine=model,
                messages=messages,
                n=1,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = response["choices"][0]["message"]["content"]
            break
        except openai.error.OpenAIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
        except openai.error.InvalidRequestError as e:
            print(type(e), e)
            break
        except KeyError:
            print(response)
            break

    return output


def chat_completion_anthropic(model, conv, temperature, max_tokens, api_dict=None):
    if api_dict is not None and "api_key" in api_dict:
        api_key = api_dict["api_key"]
    else:
        api_key = os.environ["ANTHROPIC_API_KEY"]

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            c = anthropic.Anthropic(api_key=api_key)
            prompt = [text for role, text in conv.messages if role == "Human"][0]
            response = c.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": [{"type": "text", "text": prompt}]}
                ],
            )
            output = response.content[0].text
            break
        except anthropic.APIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
    return output.strip()


def chat_completion_mistral(model, conv, temperature, max_tokens, api_dict=None):
    if api_dict is not None and "api_key" in api_dict:
        api_key = api_dict["api_key"]
    else:
        api_key = os.environ["MISTRAL_API_KEY"]

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            from mistralai.client import MistralClient
            from mistralai.models.chat_completion import ChatMessage
            
            client = MistralClient(api_key=api_key)
            prompt = prompt = conv.messages[0][1]

            chat_response = client.chat(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[ChatMessage(role="user", content=prompt)]
            )

            output = chat_response.choices[0].message.content
            break
        except Exception as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
    return output.strip()

def chat_completion_cohere(model, conv, temperature, max_tokens, api_dict=None):
    if api_dict is not None and "api_key" in api_dict:
        api_key = api_dict["api_key"]
    else:
        api_key = os.environ["CO_API_KEY"]

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            import cohere
            
            co = cohere.Client(api_key=api_key)
            prompt = prompt = [text for role, text in conv.messages if role == "user"][0]

            response = co.chat(
                model=model,
                max_tokens=min(max_tokens, 4000),
                temperature=temperature,
                message=prompt,
            )
            output = response.text
            break
        except Exception as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
    return output.strip()

def chat_completion_palm(chat_state, model, conv, temperature, max_tokens):
    from fastchat.serve.api_provider import init_palm_chat

    assert model == "palm-2-chat-bison-001"

    if chat_state is None:
        chat_state = init_palm_chat("chat-bison@001")

    parameters = {
        "temperature": temperature,
        "top_p": 0.8,
        "top_k": 40,
        "max_output_tokens": max_tokens,
    }
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = chat_state.send_message(conv.messages[-2][1], **parameters)
            output = response.text
            break
        except Exception as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
    return chat_state, output


def normalize_game_key_single(gamekey, result):
    """Make the model names sorted in a game key."""
    qid, model_1, model_2 = gamekey
    if model_1 < model_2:
        return gamekey, result
    else:
        new_gamekey = (qid, model_2, model_1)
        new_result = {
            "winners": tuple(reverse_model_map.get(x, x) for x in result["winners"]),
            "g1_judgment": result["g2_judgment"],
            "g2_judgment": result["g1_judgment"],
        }
        return new_gamekey, new_result


def normalize_game_key_dict(judgment_dict):
    """Make the model names sorted in the game keys."""
    ret = {}
    for key, value in judgment_dict.items():
        new_key, new_value = normalize_game_key_single(key, value)
        ret[new_key] = new_value
    return ret


def load_single_model_judgments(filename: str):
    """Load model judgments.

    The return value is a dict of type:
    Dict[judge: Tuple -> Dict[game_key: tuple -> game_result: dict]
    """
    judge_dict = {}

    for line in open(filename):
        obj = json.loads(line)
        judge = tuple(obj["judge"])
        qid, model = obj["question_id"], obj["model"]

        if judge not in judge_dict:
            judge_dict[judge] = {}

        gamekey = (qid, model)

        judge_dict[judge][gamekey] = {
            "score": obj["score"],
            "judgment": obj["judgment"],
        }
    return judge_dict


def check_data(questions, model_answers, models):
    # check model answers
    for m in models:
        assert m in model_answers, f"Missing model answer for {m}"
        m_answer = model_answers[m]
        for q in questions:
            assert (
                q["question_id"] in m_answer
            ), f"Missing model {m}'s answer to Question {q['question_id']}"


def get_model_list(answer_dir):
    file_paths = glob.glob(f"{answer_dir}/*.jsonl")
    file_names = [os.path.splitext(os.path.basename(f))[0] for f in file_paths]
    return file_names
