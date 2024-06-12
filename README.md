# LiveBench

Most of the code we will use for generation and evaluation is currently in `livebench/`. This is adapted from [FastChat](https://github.com/lm-sys/FastChat)'s `llm_judge` module. 

## Installation

Tested on Python 3.10

```bash
cd LiveBench
pip install torch packaging
pip install -e .
```

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

## Citation

```bibtex
@misc{livebench,
  author    = {White, Colin and Dooley, Samuel and Roberts, Manley and Pal, Arka and Feuer, Ben and Jain, Siddhartha and Shwartz-Ziv, Ravid and Jain, Neel and Saifullah, Khalid and Naidu, Siddartha and Hegde, Chinmay and LeCun, Yann and Goldstein, Tom and Neiswanger, Willie and Goldblum, Micah},
  title     = {LiveBench: A Challenging, Contamination-Free LLM Benchmark},
  url       = {https://livebench.ai},
  year      = {2024},
}```
