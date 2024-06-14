# LiveBench

![Crates.io](https://img.shields.io/crates/l/Ap?color=orange)

<p align="center">
    <a href="https://livebench.ai/">🏆 Leaderboard</a> •
    <a href="https://huggingface.co/livebench">💻 Data </a> •
    <a href="https://livebench.ai/livebench.pdf">📝 Paper</a> 
</p>

Leaderboard as of 12th June 2024:

![image](https://github.com/naszilla/LiveBench/assets/127691629/a422b0e9-1e17-4c6f-8ad3-50069e44cf0e)


## Introduction

Introducing LiveBench: a benchmark for LLMs designed with test set contamination and objective evaluation in mind.

LiveBench has the following properties:

* LiveBench is designed to limit potential contamination by releasing new questions monthly, as well as having questions based on recently-released datasets, arXiv papers, news articles, and IMDb movie synopses.
* Each question has verifiable, objective ground-truth answers, allowing hard questions to be scored accurately and automatically, without the use of an LLM judge.
* LiveBench currently contains a set of 18 diverse tasks across 6 categories, and we will release new, harder tasks over time.

## Installation Quickstart

Tested on Python 3.10

```bash
cd LiveBench
pip install torch packaging # These need to be installed prior to other dependencies.
pip install -e .
```
Note: The fastchat package version on pip is currently out of date and so we strongly recommend `pip uninstall fastchat` before running the above so that you pick up a more recent commit from the dependencies.

Note for CPU users: If installing on a CPU-only machine (e.g. to run api models only), you will need to manually remove flash-attn from the requirements list in pyproject.toml.

Our repo is adapted from FastChat's excellent [llm_judge](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) module, and it also contains code from [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) and [IFEval](https://github.com/Rohan2002/IFEval?tab=readme-ov-file).

## Usage

```bash
cd livebench
```

To generate model answers on LiveBench, run:
```bash
python gen_model_answer.py --model-path /path/to/Mistral-7B-Instruct-v0.2/ --model-id Mistral-7B-Instruct-v0.2 --dtype bfloat16 --bench-name live_bench
```

For API-based models, first set the appropriate key and then run the `gen_api_answer.py`. We currently support the following APIs: OpenAI, Anthropic, Mistral, Cohere, and Gemini. The command to run all of LiveBench on an `api_model_name`, run this command:
```bash
export OPENAI_API_KEY=<your_key>
export ANTHROPIC_API_KEY=<your_key>
export MISTRAL_API_KEY=<your_key>
export CO_API_KEY=<your_key>
export GEMINI_API_KEY=<your_key>
python gen_api_answer.py --model <api_model_name> --bench-name live_bench
```

To generate model answers with VLLM or other arbitrary APIs matching the OpenAI API format, run:
```bash
export LIVEBENCH_API_KEY=<your API key if needed. Usually not needed for VLLM>
python gen_api_answer.py --model <api_model_name> --bench-name live_bench --api-base <your endpoint. Often, for VLLM, this is http://localhost:8000/v1>
```

To score the model outputs:

```bash
python gen_ground_truth_judgment.py --bench-name live_bench
```

To show all the results:
```bash
python show_livebench_results.py
```

You may want to run these commands on just some models. To run any of the above python files (`gen_model_answer.py`, `gen_api_answer.py`, `gen_ground_truth_judgment`, or `show_livebench_results`) for specific models, use the following argument styles:
```bash
python gen_model_answer.py          --bench-name live_bench --model-path /path/to/Mistral-7B-Instruct-v0.2/ --model-id Mistral-7B-Instruct-v0.2 --dtype bfloat16 
python gen_api_answer.py            --bench-name live_bench --model gpt-4-turbo
python gen_ground_truth_judgment.py --bench-name live_bench --model-list Mistral-7B-Instruct-v0.2 Llama-2-7b-chat-hf claude-3-opus-20240229
python show_livebench_results.py    --bench-name live_bench --model-list Mistral-7B-Instruct-v0.2 Llama-2-7b-chat-hf claude-3-opus-20240229
```

Or, you may want to show results for a specific category or task of LiveBench by using the `--bench-name` argument. Here, we run `show_livebench_results.py` on just the `web_of_lies_v2` task: 
```bash
python show_livebench_results.py --bench-name live_bench/reasoning/web_of_lies_v2
```

To optionally download `question.jsonl` files (for inspection) and answer/judgment files from the leaderboard, use
```bash
python download_questions.py
python download_leaderboard.py
```

Also present is some modified and unmodified code from https://github.com/LiveCodeBench/LiveCodeBench.

## Data
The questions for each of the categories can be found below:
- [Reasoning](https://huggingface.co/datasets/livebench/reasoning)
- [Math](https://huggingface.co/datasets/livebench/math)
- [Coding](https://huggingface.co/datasets/livebench/coding)
- [Language](https://huggingface.co/datasets/livebench/language)
- [Data Analysis](https://huggingface.co/datasets/livebench/data_analysis)
- [Instruction Following](https://huggingface.co/datasets/livebench/instruction_following)

Also available are the [model answers](https://huggingface.co/datasets/livebench/model_answer) and the [model judgments](https://huggingface.co/datasets/livebench/model_judgment).

## Citation

```bibtex
@misc{livebench,
  author    = {White, Colin and Dooley, Samuel and Roberts, Manley and Pal, Arka and Feuer, Ben and Jain, Siddhartha and Shwartz-Ziv, Ravid and Jain, Neel and Saifullah, Khalid and Naidu, Siddartha and Hegde, Chinmay and LeCun, Yann and Goldstein, Tom and Neiswanger, Willie and Goldblum, Micah},
  title     = {LiveBench: A Challenging, Contamination-Free LLM Benchmark},
  url       = {https://livebench.ai},
  year      = {2024},
}```
