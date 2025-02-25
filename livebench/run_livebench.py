#!/usr/bin/env python3
"""
Run LiveBench benchmarks with various execution modes and options.
This script consolidates the functionality of the bash scripts into a single Python interface.
"""

import argparse
import os
import sys
import time
import libtmux
import subprocess
from typing import List, Optional, Union

# Default benchmarks used when none specified
DEFAULT_BENCHMARKS = [
    "live_bench/coding",
    "live_bench/data_analysis", 
    "live_bench/instruction_following",
    "live_bench/language",
    "live_bench/math",
    "live_bench/reasoning"
]

def run_command(cmd: str, env: Optional[dict] = None) -> int:
    """Run a shell command and return its exit code"""
    try:
        print(f"Running: {cmd}")
        result = subprocess.run(["bash", "-c", cmd], env=env, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(f"Exit code: {e.returncode}")
        return e.returncode

def setup_tmux_session(session_name: str, benchmarks: List[str], commands: List[str], venv_path: Optional[str] = None) -> None:
    """
    Set up a tmux session with panes for each benchmark.
    
    Args:
        session_name: Name for the tmux session
        benchmarks: List of benchmark paths to run
        commands: List of commands to run for each benchmark
        venv_path: Optional path to virtual environment to activate
    """
    print(f"\nSetting up tmux session '{session_name}' for benchmarks: {', '.join(benchmarks)}")
    
    # Initialize tmux server
    server = libtmux.Server()
    
    # Kill existing session if it exists
    try:
        existing_sessions = [s for s in server.sessions if s.name == session_name]
        if existing_sessions:
            print(f"Killing existing session '{session_name}'")
            existing_sessions[0].kill()
    except:
        pass

    # Create new session
    print("Creating new tmux session")
    session = server.new_session(session_name=session_name)
    window = session.active_window

    # Create panes one at a time
    print("Creating panes...")
    panes = [window.panes[0]]  # Start with the initial pane
    
    # Create additional panes as needed
    for i in range(1, len(benchmarks)):
        try:
            # Alternate between vertical and horizontal splits
            new_pane = panes[-1].split('-h' if i % 2 == 0 else '-v')
            panes.append(new_pane)
            # Try to even out the layout after each split
            window.select_layout('tiled')
        except Exception as e:
            print(f"Warning: Could not create all requested panes. Will continue with {len(panes)} panes.")
            print(f"Error: {str(e)}")
            break

    # Distribute commands across available panes
    print("Setting up panes...")
    for i, (pane, benchmark, cmd) in enumerate(zip(panes, benchmarks, commands)):
        print(f"Setting up pane {i+1} for benchmark: {benchmark}")
        
        # Activate virtualenv if provided
        if venv_path:
            pane.send_keys(f"source {venv_path}")
            time.sleep(0.5)
        
        # Run the command
        pane.send_keys(cmd)
        time.sleep(0.5)
    
    # Final layout adjustment
    print("Adjusting final layout...")
    window.select_layout('tiled')

    # If we couldn't create enough panes, warn about the remaining benchmarks
    if len(panes) < len(benchmarks):
        remaining = benchmarks[len(panes):]
        print(f"\nWarning: Could not create panes for the following benchmarks due to space constraints:")
        for bench in remaining:
            print(f"  - {bench}")
        print("\nSuggestion: Consider running these benchmarks in sequential mode instead.")

def build_run_command(
    model: str,
    mode: str,
    venv: Optional[str] = None,
    bench_name: Optional[Union[str, List[str]]] = None,
    question_source: str = "huggingface",
    api_base: Optional[str] = None,
    api_key_name: Optional[str] = None,
    model_display_name: Optional[str] = None,
    max_tokens: Optional[int] = None,
    parallel_requests: Optional[int] = None,
    resume: bool = False,
    retry_failures: bool = False,
    skip_inference: bool = False,
    skip_grading: bool = False
) -> str:
    """Build the command to run gen_api_answer and gen_ground_truth_judgment in sequence"""
    
    # Build gen_api_answer command
    gen_api_cmd = f"python -u gen_api_answer.py --model {model} --question-source {question_source}"
    
    # Build gen_ground_truth_judgment command
    gen_judge_cmd = f"python -u gen_ground_truth_judgment.py --model {model} --question-source {question_source}"
    
    # Add bench_name to both commands
    if isinstance(bench_name, list):
        bench_str = ' '.join(bench_name)
        gen_api_cmd += f" --bench-name {bench_str}"
        gen_judge_cmd += f" --bench-name {bench_str}"
    elif bench_name:
        gen_api_cmd += f" --bench-name {bench_name}"
        gen_judge_cmd += f" --bench-name {bench_name}"
    
    # Add optional arguments to gen_api_answer
    if api_base:
        gen_api_cmd += f" --api-base {api_base}"
    if api_key_name:
        gen_api_cmd = f"export LIVEBENCH_API_KEY=${api_key_name} && {gen_api_cmd}"
    if model_display_name:
        gen_api_cmd += f" --model-display-name {model_display_name}"
        gen_judge_cmd += f" --model-display-name {model_display_name}"
    if max_tokens:
        gen_api_cmd += f" --max-tokens {max_tokens}"
    if parallel_requests:
        gen_api_cmd += f" --parallel {parallel_requests}"
    if resume:
        gen_api_cmd += " --resume"
    if retry_failures:
        gen_api_cmd += " --retry-failures"
    
    # Chain the commands together with && to ensure they run in sequence
    # Only run gen_ground_truth_judgment if gen_api_answer succeeds
    if skip_inference and not skip_grading:
        return gen_judge_cmd
    elif skip_grading and not skip_inference:
        return gen_api_cmd
    elif skip_inference and skip_grading:
        return "echo 'Both inference and grading are skipped'"
    else:
        return f"{gen_api_cmd} && {gen_judge_cmd}"

def run_model(
    model: str,
    mode: str,
    venv: Optional[str] = None,
    bench_names: Optional[List[str]] = None,
    question_source: str = "huggingface",
    api_base: Optional[str] = None,
    api_key_name: Optional[str] = None,
    model_display_name: Optional[str] = None,
    max_tokens: Optional[int] = None,
    parallel_requests: Optional[int] = None,
    skip_inference: bool = False,
    skip_grading: bool = False,
    resume: bool = False,
    retry_failures: bool = False
) -> None:
    """Run livebench for a single model"""
    if mode == "parallel":
        run_parallel(
            model=model,
            venv=venv,
            bench_names=bench_names,
            question_source=question_source,
            api_base=api_base,
            api_key_name=api_key_name,
            model_display_name=model_display_name,
            parallel_requests=parallel_requests
        )
    elif mode == "sequential":
        run_sequential(
            model=model,
            venv=venv,
            bench_names=bench_names,
            question_source=question_source,
            api_base=api_base,
            api_key_name=api_key_name,
            model_display_name=model_display_name,
            max_tokens=max_tokens
        )
    else:  # single mode
        if not bench_names:
            bench_names = ["live_bench"]
        for bench in bench_names:
            run_single(
                model=model,
                bench_name=bench,
                venv=venv,
                question_source=question_source,
                api_base=api_base,
                api_key_name=api_key_name,
                model_display_name=model_display_name,
                max_tokens=max_tokens,
                parallel_requests=parallel_requests,
                resume=resume,
                retry_failures=retry_failures,
                skip_inference=skip_inference,
                skip_grading=skip_grading
            )

def run_sequential(
    model: str,
    venv: Optional[str] = None,
    bench_names: Optional[List[str]] = None,
    question_source: str = "huggingface",
    api_base: Optional[str] = None,
    api_key_name: Optional[str] = None,
    model_display_name: Optional[str] = None,
    max_tokens: Optional[int] = None
) -> None:
    """Run benchmarks sequentially in a single tmux session"""
    print(f"\nRunning sequential benchmarks for model: {model}")
    session_name = f"livebench-{model}".replace(".", "_").replace(":", "_")
    
    # If no bench_names provided, run all benchmarks in sequence using live_bench
    if not bench_names:
        print("No specific benchmarks provided, running all benchmarks in sequence")
        cmd = build_run_command(
            model=model,
            mode="single",
            venv=venv,
            bench_name="live_bench",
            question_source=question_source,
            api_base=api_base,
            api_key_name=api_key_name,
            model_display_name=model_display_name,
            max_tokens=max_tokens,
            resume=True,
            retry_failures=True
        )
        setup_tmux_session(session_name, ["live_bench"], [cmd], venv)
        return
    
    print(f"Running benchmarks sequentially: {', '.join(bench_names)}")
    # Build commands for each benchmark
    print("Building commands for each benchmark...")
    commands = []
    for bench in bench_names:
        cmd = build_run_command(
            model=model,
            mode="single",
            venv=venv,
            bench_name=bench,
            question_source=question_source,
            api_base=api_base,
            api_key_name=api_key_name,
            model_display_name=model_display_name,
            max_tokens=max_tokens,
            resume=True,
            retry_failures=True
        )
        commands.append(cmd)
    
    # Join commands with semicolons for sequential execution
    full_cmd = " ; ".join(commands)
    
    # Set up tmux session
    setup_tmux_session(session_name, [bench_names[0]], [full_cmd], venv)

def run_parallel(
    model: str,
    venv: Optional[str] = None,
    bench_names: Optional[List[str]] = None,
    question_source: str = "huggingface",
    api_base: Optional[str] = None,
    api_key_name: Optional[str] = None,
    model_display_name: Optional[str] = None,
    parallel_requests: Optional[int] = None
) -> None:
    """Run benchmarks in parallel in separate tmux panes"""
    print(f"\nRunning parallel benchmarks for model: {model}")
    session_name = f"livebench-{model}".replace(".", "_").replace(":", "_")
    
    # If no bench_names provided, use DEFAULT_BENCHMARKS for parallelization
    benchmarks = bench_names if bench_names else DEFAULT_BENCHMARKS
    print(f"Using benchmarks: {', '.join(benchmarks)}")
    
    # Build commands for each benchmark
    print("Building commands for each benchmark...")
    commands = []
    for bench in benchmarks:
        cmd = build_run_command(
            model=model,
            mode="single",
            venv=venv,
            bench_name=bench,
            question_source=question_source,
            api_base=api_base,
            api_key_name=api_key_name,
            model_display_name=model_display_name,
            parallel_requests=parallel_requests
        )
        commands.append(cmd)
    
    # Set up tmux session with parallel panes
    setup_tmux_session(session_name, benchmarks, commands, venv)

def run_single(
    model: str,
    bench_name: str,
    venv: Optional[str] = None,
    question_source: str = "huggingface",
    api_base: Optional[str] = None,
    api_key_name: Optional[str] = None,
    model_display_name: Optional[str] = None,
    max_tokens: Optional[int] = None,
    parallel_requests: Optional[int] = None,
    resume: bool = False,
    retry_failures: bool = False,
    skip_inference: bool = False,
    skip_grading: bool = False
) -> int:
    """
    Run a single benchmark.
    
    Args:
        model: Model identifier (e.g., gpt-4)
        bench_name: Benchmark path
        venv: Optional path to virtual environment
        question_source: Source of questions (huggingface/jsonl)
        api_base: Base URL for API requests
        api_key_name: Environment variable name for API key
        model_display_name: Display name in results
        max_tokens: Maximum tokens for responses
        parallel_requests: Number of parallel API requests
        resume: Whether to resume from previous run
        retry_failures: Whether to retry failed generations
        skip_inference: Whether to skip running gen_api_answer.py
        skip_grading: Whether to skip running gen_ground_truth_judgment.py
    
    Returns:
        Exit code from the last command executed
    """
    print(f"\nRunning single benchmark '{bench_name}' for model: {model}")
    
    # Build the chained command using build_run_command
    cmd = build_run_command(
        model=model,
        mode="single",
        venv=venv,
        bench_name=bench_name,
        question_source=question_source,
        api_base=api_base,
        api_key_name=api_key_name,
        model_display_name=model_display_name,
        max_tokens=max_tokens,
        parallel_requests=parallel_requests,
        resume=resume,
        retry_failures=retry_failures,
        skip_inference=skip_inference,
        skip_grading=skip_grading
    )
    
    # Run the command
    if venv:
        print(f"Activating virtual environment: {venv}")
        os.environ["PATH"] = f"{os.path.dirname(venv)}:{os.environ['PATH']}"
        run_command(f"source {venv}")
    
    print("\nRunning benchmark command:")
    print(cmd)
    exit_code = run_command(cmd)
    
    if exit_code != 0:
        print("Benchmark failed!")
    else:
        print("Benchmark completed successfully!")
    return exit_code

def main():
    parser = argparse.ArgumentParser(description="Run LiveBench benchmarks with various execution modes")
    
    # Required arguments
    parser.add_argument("--model", required=True, nargs="+", help="One or more model identifiers (e.g., gpt-4)")
    
    # Optional arguments
    parser.add_argument("--venv", help="Path to virtual environment to activate")
    parser.add_argument("--mode", choices=["single", "parallel", "sequential"], default="single",
                      help="Execution mode: single benchmark, parallel benchmarks, or sequential benchmarks")
    parser.add_argument("--bench-name", nargs="+",
                      help="One or more benchmark paths to run. If not provided: for single/sequential modes, "
                           "runs all benchmarks in sequence using 'live_bench'. For parallel mode, uses "
                           "predefined benchmark list for parallelization.")
    parser.add_argument("--question-source", default="huggingface", choices=["huggingface", "jsonl"],
                      help="Source of benchmark questions")
    parser.add_argument("--api-base", help="Base URL for API requests")
    parser.add_argument("--api-key-name", help="Environment variable name containing the API key")
    parser.add_argument("--model-display-name", help="Display name for the model in results")
    parser.add_argument("--max-tokens", type=int, help="Maximum tokens for model responses")
    parser.add_argument("--parallel-requests", type=int, help="Number of parallel requests for API calls")
    parser.add_argument("--resume", action="store_true", help="Resume from previous run")
    parser.add_argument("--retry-failures", action="store_true", help="Retry failed generations")
    parser.add_argument("--skip-inference", action="store_true", help="Skip running gen_api_answer.py")
    parser.add_argument("--skip-grading", action="store_true", help="Skip running gen_ground_truth_judgment.py")
    
    args = parser.parse_args()
    
    print("\nStarting LiveBench evaluation")
    print(f"Mode: {args.mode}")
    print(f"Models: {', '.join(args.model)}")
    if args.bench_name:
        print(f"Benchmarks: {', '.join(args.bench_name)}")
    print(f"Question source: {args.question_source}")
    
    # Run each model in its own tmux session
    for model in args.model:
        run_model(
            model=model,
            mode=args.mode,
            venv=args.venv,
            bench_names=args.bench_name,
            question_source=args.question_source,
            api_base=args.api_base,
            api_key_name=args.api_key_name,
            model_display_name=args.model_display_name,
            max_tokens=args.max_tokens,
            parallel_requests=args.parallel_requests,
            resume=args.resume,
            retry_failures=args.retry_failures,
            skip_inference=args.skip_inference,
            skip_grading=args.skip_grading
        )

if __name__ == "__main__":
    main()