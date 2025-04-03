# LiveBench

![Crates.io](https://img.shields.io/crates/l/Ap?color=orange)

<p align="center">
    <a href="https://livebench.ai/">🏆 Leaderboard</a> •
    <a href="https://huggingface.co/livebench">💻 Data </a> •
    <a href="https://arxiv.org/abs/2406.19314">📝 Paper</a> 
</p>

LiveBench is a cutting-edge benchmark suite designed to evaluate Large Language Models (LLMs) with a focus on preventing test set contamination while ensuring objective and accurate evaluation. By releasing new questions monthly and incorporating recently-released datasets, LiveBench provides a dynamic and challenging environment for assessing LLM capabilities across diverse tasks.

**[Spotlight Paper at ICLR 2025](https://openreview.net/forum?id=sKYHBTAxVa)**

## Latest Results

Top models as of 30th September 2024 (for a full up-to-date leaderboard, see [livebench.ai](https://livebench.ai/)):

![image](assets/livebench-2024-09-30.png)

Please see the [changelog](changelog.md) for details about each LiveBench release.

## Key Features

- **Contamination-Free Design**: Monthly release of new questions based on recent datasets, papers, news, and media
- **Objective Evaluation**: Verifiable ground-truth answers enable accurate scoring without LLM judges
- **Diverse Task Categories**: 18 tasks across 6 categories:
  - Reasoning
  - Math
  - Coding
  - Language
  - Data Analysis
  - Instruction Following
- **Continuous Evolution**: Regular addition of new, challenging tasks
- **Open Participation**: We welcome and evaluate new models! [Open an issue](https://github.com/LiveBench/LiveBench/issues) or email [livebench.ai@gmail.com](mailto:livebench.ai@gmail.com)

## Quick Start

1. **Setup Environment**
```bash
python -m venv .venv
source .venv/bin/activate
cd LiveBench
```

2. **Install LiveBench**
```bash
# For API models and basic usage:
pip install -e .

# For local GPU inference support:
pip install -e .[flash_attn]
```

3. **Run Evaluation**
```bash
# Basic usage with API model
python run_livebench.py --model gpt-4o --bench-name live_bench/coding

# View results
python show_livebench_result.py --bench-name live_bench/coding --model-list gpt-4o
```

## Detailed Installation Guide

### Prerequisites
- Python 3.10
- Virtual environment (recommended)
- For local GPU inference: CUDA-compatible GPU

### Important Notes
- **FastChat Package**: The pip version of `fschat` may be outdated. Run `pip uninstall fschat` before installation to ensure the correct version is installed.
- **Dependencies**: The installation includes all necessary dependencies for your chosen setup (API-only or full GPU support).
- **GPU Support**: The `flash_attn` extra is only needed if you plan to run models locally on GPU.

## Usage Guide

### Running Evaluations

#### Basic Usage
```bash
python run_livebench.py --model <model-name> --bench-name <benchmark>
```

#### Common Options
- `--bench-name`: Select benchmark subset (e.g., `live_bench/coding`)
- `--model`: Model to evaluate
- `--max-tokens`: Response token limit
- `--api-base`: Custom API endpoint
- `--api-key-name`: API key environment variable
- `--parallel-requests`: Concurrent API request limit
- `--resume`: Continue interrupted runs
- `--retry-failures`: Retry failed questions

### Parallel Evaluation Strategies

#### High Throughput Setup
For APIs with high rate limits:
```bash
python run_livebench.py --model gpt-4o --bench-name live_bench --mode parallel --parallel-requests 10
```

#### Limited Rate Setup
For APIs with lower limits:
```bash
python run_livebench.py --model claude-3-5-sonnet --bench-name live_bench --mode parallel --parallel-requests 2
```

### Viewing Results

Display evaluation results with detailed breakdowns:
```bash
# Show specific models
python show_livebench_result.py --bench-name live_bench/coding --model-list gpt-4o claude-3-5-sonnet

# Compare across categories
python show_livebench_result.py --bench-name live_bench/coding live_bench/math --model-list gpt-4o
```

Results are saved in:
- `all_groups.csv`: Category-wise breakdown
- `all_tasks.csv`: Task-wise breakdown

## Troubleshooting

### Common Issues

1. **API Errors**
   - Check rate limits and adjust `--parallel-requests`
   - Verify API key and endpoint configuration
   - Use `--mode single` for more stable execution

2. **Content Filter Blocks**
   - Some models (especially Gemini) may block certain questions
   - Failed responses are marked as incorrect
   - Use `scripts/error_check` to identify blocked questions

3. **Resource Usage**
   - Monitor GPU memory for local inference
   - Adjust batch sizes if needed
   - Consider using API endpoints for large models

### Error Recovery

1. Check failed questions:
```bash
python scripts/error_check
```

2. Retry failed evaluations:
```bash
python scripts/rerun_failed_questions.py
# or
python run_livebench.py --resume --retry-failures
```

## Data Access

### Question Datasets
Access task-specific questions on HuggingFace:
- [Reasoning](https://huggingface.co/datasets/livebench/reasoning)
- [Math](https://huggingface.co/datasets/livebench/math)
- [Coding](https://huggingface.co/datasets/livebench/coding)
- [Language](https://huggingface.co/datasets/livebench/language)
- [Data Analysis](https://huggingface.co/datasets/livebench/data_analysis)
- [Instruction Following](https://huggingface.co/datasets/livebench/instruction_following)

Additional resources:
- [Model Answers](https://huggingface.co/datasets/livebench/model_answer)
- [Model Judgments](https://huggingface.co/datasets/livebench/model_judgment)

### Download Data
```bash
# Get questions
python download_questions.py

# Get leaderboard data
python download_leaderboard.py
```

## Extending LiveBench

### Adding New Questions

1. Create `question.jsonl` in appropriate directory:
```
livebench/data/live_bench/<category>/<task>/question.jsonl
```

2. Implement scoring method in `process_results` folder
3. Update `gen_ground_truth_judgment.py`
4. Test with:
```bash
python gen_api_answer.py --bench-name live_bench/<category>/<task> --model <model> --question-source jsonl
python gen_ground_truth_judgment.py --bench-name live_bench/<category>/<task> --question-source jsonl
python show_livebench_result.py --bench-name live_bench/<category>/<task>
```

### Adding New Models

#### For OpenAI-Compatible APIs
Use `run_livebench.py` with `--api-base` parameter

#### For Other APIs
1. Implement completion function in `model/completions.py`
2. Add model adapter in `model/model_adapter.py` if needed
3. Add model entry in `model/api_models.py`

## Documentation

Detailed documentation available in:
- [Author Responsibility](docs/AUTHOR_RESPONSIBILITY.md)
- [Code of Conduct](docs/CODE_OF_CONDUCT.md)
- [Contributing](docs/CONTRIBUTING.md)
- [Datasheet](docs/DATASHEET.md)
- [Maintenance Plan](docs/MAINTENANCE_PLAN.md)

## Citation

```bibtex
@inproceedings{livebench,
  title={LiveBench: A Challenging, Contamination-Free {LLM} Benchmark},
  author={Colin White and Samuel Dooley and Manley Roberts and Arka Pal and Benjamin Feuer and Siddhartha Jain and Ravid Shwartz-Ziv and Neel Jain and Khalid Saifullah and Sreemanti Dey and Shubh-Agrawal and Sandeep Singh Sandha and Siddartha Venkat Naidu and Chinmay Hegde and Yann LeCun and Tom Goldstein and Willie Neiswanger and Micah Goldblum},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
}
```