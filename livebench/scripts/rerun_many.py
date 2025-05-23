#!/usr/bin/env python3

import subprocess
import sys
import time
import logging
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('rerun_many.log')
    ]
)
logger = logging.getLogger(__name__)

# List of models to evaluate
MODELS = [
    "o3-medium",
    "o4-mini-medium",
    "deepseek-r1",
    "gemini-2.5-flash-preview",
  #  "grok-3-beta",                # grok 3 beta
    "claude-3-7-sonnet",
     "gpt-4.1",
    "claude-3-5-sonnet", # claude 3.5 sonnet
    "gpt-4.1-mini",           # gpt 4.1 mini
    "llama4-maverick",
  #  "grok-3-mini-beta-high",
    "claude-3-5-haiku", # claude 3.5 haiku
    "chatgpt-4o-latest",         # chatgpt 4o latest
    "deepseek-v3-0324",               # deepseek v3
    "gpt-4o",                    # gpt 4o
    "qwen-2.5-72b-instruct-turbo",              # qwen 2.5 72b
    "gpt-4.1-nano",
    "deepseek-r1-distill-llama-70b",
    "deepseek-r1-distill-qwen-32b",
    "qwen3-235b-a22b-thinking",
    "qwen3-30b-a3b-thinking",
    "qwen3-32b-thinking"
]


def run_benchmark(model_name: str) -> None:
    """
    Run the LiveBench benchmark for a specific model.
    
    Args:
        model_name: The name of the model to evaluate
    """
    logger.info(f"Starting benchmark for model: {model_name}")
    
    cmd = [
        "python", "run_livebench.py",
        "--bench-name", "live_bench/coding/agentic_coding",
        "--question-source", "jsonl",
        "--parallel-requests", "35",
        "--parallel-grading", "35",
        "--resume",
        "--retry-failures",
        "--model", model_name
    ]
    
    try:
        start_time = time.time()
        result = subprocess.run(
            cmd,
            check=True,
            text=True
        )
        elapsed_time = time.time() - start_time
        
        logger.info(f"Benchmark for {model_name} completed sudccessfully in {elapsed_time:.2f} seconds")
        logger.debug(f"Output: {result.stdout}")
        
        if result.stderr:
            logger.warning(f"Stderr for {model_name}: {result.stderr}")
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Benchmark for {model_name} failed with exit code {e.returncode}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
    except Exception as e:
        logger.error(f"Unexpected error running benchmark for {model_name}: {str(e)}")


def main() -> None:
    """
    Main function to run benchmarks for all models in sequence.
    """
    logger.info(f"Starting benchmark run for {len(MODELS)} models")
    
    for i, model in enumerate(MODELS, 1):
        logger.info(f"Running model {i}/{len(MODELS)}: {model}")
        run_benchmark(model)
        logger.info(f"Completed model {i}/{len(MODELS)}: {model}")
    
    logger.info("All benchmark runs completed")


if __name__ == "__main__":
    main()
