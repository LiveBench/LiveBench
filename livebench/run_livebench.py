#!/usr/bin/env python3
"""
Run LiveBench benchmarks with various execution modes and options.
This script consolidates the functionality of the bash scripts into a single Python interface.
"""

import argparse
import os
import time
import libtmux
import subprocess
from dataclasses import dataclass
import dotenv

dotenv.load_dotenv()

# Default benchmarks used when none specified
DEFAULT_BENCHMARKS = [
    "live_bench/coding",
    "live_bench/data_analysis", 
    "live_bench/instruction_following",
    "live_bench/language",
    "live_bench/math",
    "live_bench/reasoning",
]

# Detect if we're in an active virtual environment
def detect_active_venv():
    """Detect if a virtual environment is active and return its activate script path"""
    venv_path = os.environ.get("VIRTUAL_ENV")
    if venv_path:
        # Return the path to the activate script
        return os.path.join(venv_path, "bin", "activate")
    return None

@dataclass
class LiveBenchParams:
    """Parameters for LiveBench execution"""
    model: str | None = None
    mode: str = "single"
    venv: str | None = None
    bench_names: list[str] | None = None
    question_source: str = "huggingface"
    api_base: str | None = None
    api_key_name: str | None = None
    api_key: str | None = None
    model_display_name: str | None = None
    max_tokens: int | None = None
    parallel_requests: int | None = None
    parallel_grading: int | None = None
    resume: bool = False
    resume_inference: bool = False
    resume_grading: bool = False
    retry_failures: bool = False
    skip_inference: bool = False
    skip_grading: bool = False
    force_temperature: float | None = None
    num_choices: int | None = None
    question_begin: int | None = None
    question_end: int | None = None
    question_id: list[str] | None = None
    livebench_release_option: str | None = None
    stream: bool = False
    remove_existing_judgment_file: bool = False
    ignore_missing_answers: bool = False
    debug: bool = False
    model_provider_override: str | None = None
    only_incorrect: bool = False

    @classmethod
    def from_args(cls, args, model: str | None = None):
        """
        Create a LiveBenchParams instance from parsed command-line arguments
        
        Args:
            args: The parsed command-line arguments
            model: Optional model name to use instead of the one in args
        """
        return cls(
            model=model if model is not None else (args.model[0] if isinstance(args.model, list) else args.model),
            mode=args.mode,
            venv=args.venv,
            bench_names=args.bench_name,
            question_source=args.question_source,
            api_base=args.api_base,
            api_key_name=args.api_key_name,
            api_key=args.api_key,
            model_display_name=args.model_display_name,
            max_tokens=args.max_tokens,
            parallel_requests=args.parallel_requests,
            parallel_grading=args.parallel_grading,
            resume=args.resume,
            resume_inference=args.resume_inference,
            resume_grading=args.resume_grading,
            retry_failures=args.retry_failures,
            skip_inference=args.skip_inference,
            skip_grading=args.skip_grading,
            force_temperature=args.force_temperature,
            num_choices=args.num_choices,
            question_begin=args.question_begin,
            question_end=args.question_end,
            question_id=args.question_id,
            livebench_release_option=args.livebench_release_option,
            stream=args.stream,
            remove_existing_judgment_file=args.remove_existing_judgment_file,
            ignore_missing_answers=args.ignore_missing_answers,
            debug=args.debug,
            model_provider_override=args.model_provider_override,
            only_incorrect=args.only_incorrect
        )

def run_command(cmd: str, env: dict[str, str] | None = None) -> int:
    """Run a shell command and return its exit code"""
    try:
        print(f"Running: {cmd}")
        result = subprocess.run(["bash", "-c", cmd], env=env, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(f"Exit code: {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Stopping command and continuing to collect results...")
        return 1

def setup_tmux_session(session_name: str, benchmarks: list[str], commands: list[str], venv_path: str | None = None) -> None:
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
            if not os.path.exists(venv_path):
                print(f"Virtual environment not found at {venv_path}, skipping activation")
            else:
                print(f"Activating virtual environment: {venv_path}")
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
    else:
        print("Successfully created all panes!")
        print("Run 'tmux a' to attach to the session")

def build_run_command(
    model: str,
    bench_name: str | list[str] | None = None,
    question_source: str = "huggingface",
    api_base: str | None = None,
    api_key_name: str | None = None,
    api_key: str | None = None,
    model_display_name: str | None = None,
    max_tokens: int | None = None,
    parallel_requests: int | None = None,
    parallel_grading: int | None = None,
    resume: bool = False,
    resume_inference: bool = False,
    resume_grading: bool = False,
    retry_failures: bool = False,
    skip_inference: bool = False,
    skip_grading: bool = False,
    force_temperature: float | None = None,
    num_choices: int | None = None,
    question_begin: int | None = None,
    question_end: int | None = None,
    question_id: list[str] | None = None,
    livebench_release_option: str | None = None,
    stream: bool = False,
    remove_existing_judgment_file: bool = False,
    ignore_missing_answers: bool = False,
    debug: bool = False,
    model_provider_override: str | None = None,
    only_incorrect: bool = False
) -> str:
    """Build the command to run gen_api_answer and gen_ground_truth_judgment in sequence"""
    
    # Build gen_api_answer command
    gen_api_cmd = f"python -u gen_api_answer.py --model {model} --question-source {question_source}"
    
    # Build gen_ground_truth_judgment command
    gen_judge_cmd = f"python -u gen_ground_truth_judgment.py --question-source {question_source}"
    if model:
        gen_judge_cmd += f" --model {model}"
    
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
        gen_api_cmd = f"export LIVEBENCH_API_KEY={os.environ[api_key_name]} && {gen_api_cmd}"
    elif api_key:
        gen_api_cmd = f"export LIVEBENCH_API_KEY='{api_key}' && {gen_api_cmd}"
    if model_display_name:
        gen_api_cmd += f" --model-display-name {model_display_name}"
        gen_judge_cmd += f" --model-display-name {model_display_name}"
    if max_tokens:
        gen_api_cmd += f" --max-tokens {max_tokens}"
    if parallel_requests:
        gen_api_cmd += f" --parallel {parallel_requests}"
    if parallel_grading:
        gen_judge_cmd += f" --parallel {parallel_grading}"
    # Handle resume flags
    if resume:
        gen_api_cmd += " --resume"
        gen_judge_cmd += " --resume"
    elif resume_inference:
        gen_api_cmd += " --resume"
    elif resume_grading:
        gen_judge_cmd += " --resume"
        
    if retry_failures:
        gen_api_cmd += " --retry-failures"
    
    # Add new optional arguments
    if force_temperature is not None:
        gen_api_cmd += f" --force-temperature {force_temperature}"
    if num_choices:
        gen_api_cmd += f" --num-choices {num_choices}"
    if question_begin is not None:
        gen_api_cmd += f" --question-begin {question_begin}"
        gen_judge_cmd += f" --question-begin {question_begin}"
    if question_end is not None:
        gen_api_cmd += f" --question-end {question_end}"
        gen_judge_cmd += f" --question-end {question_end}"
    if question_id:
        question_id_str = ' '.join(question_id)
        gen_api_cmd += f" --question-id {question_id_str}"
        gen_judge_cmd += f" --question-id {question_id_str}"
    if livebench_release_option:
        gen_api_cmd += f" --livebench-release-option {livebench_release_option}"
        gen_judge_cmd += f" --livebench-release-option {livebench_release_option}"
    if stream:
        gen_api_cmd += " --stream"
    if remove_existing_judgment_file:
        gen_judge_cmd += " --remove-existing-file"
    if ignore_missing_answers:
        gen_judge_cmd += " --ignore-missing-answers"
    if model_provider_override:
        gen_api_cmd += f" --model-provider-override {model_provider_override}"
    if only_incorrect:
        gen_judge_cmd += " --only-incorrect"

    # Add debug flag only to judgment command
    if debug:
        gen_judge_cmd += " --debug"
    
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

def build_run_command_from_params(params: LiveBenchParams, bench_name: str | list[str] | None = None) -> str:
    """Build the command to run gen_api_answer and gen_ground_truth_judgment using LiveBenchParams"""
    return build_run_command(
        model=params.model if params.model is not None else "",
        bench_name=bench_name if bench_name is not None else params.bench_names,
        question_source=params.question_source,
        api_base=params.api_base,
        api_key_name=params.api_key_name,
        api_key=params.api_key,
        model_display_name=params.model_display_name,
        max_tokens=params.max_tokens,
        parallel_requests=params.parallel_requests,
        parallel_grading=params.parallel_grading,
        resume=params.resume,
        resume_inference=params.resume_inference,
        resume_grading=params.resume_grading,
        retry_failures=params.retry_failures,
        skip_inference=params.skip_inference,
        skip_grading=params.skip_grading,
        force_temperature=params.force_temperature,
        num_choices=params.num_choices,
        question_begin=params.question_begin,
        question_end=params.question_end,
        question_id=params.question_id,
        livebench_release_option=params.livebench_release_option,
        stream=params.stream,
        remove_existing_judgment_file=params.remove_existing_judgment_file,
        only_incorrect=params.only_incorrect,
        ignore_missing_answers=params.ignore_missing_answers,
        debug=params.debug,
        model_provider_override=params.model_provider_override
    )

def run_model(params: LiveBenchParams) -> None:
    """Run livebench for a single model"""
    if params.mode == "parallel":
        run_parallel(params)
    elif params.mode == "sequential":
        run_sequential(params)
    else:  # single mode
        run_single(params)

def run_sequential(params: LiveBenchParams) -> None:
    """Run benchmarks together in a single tmux session"""
    print(f"\nRunning benchmarks for model: {params.model}")
    session_name = f"livebench-{params.model}".replace(".", "_").replace(":", "_")
    
    # If no bench_names provided, run all benchmarks in sequence using live_bench
    if not params.bench_names:
        print("No specific benchmarks provided, running all benchmarks in sequence")
        cmd = build_run_command_from_params(params, bench_name=DEFAULT_BENCHMARKS)
        setup_tmux_session(session_name, ["live_bench"], [cmd], params.venv)
        return
    elif params.bench_names == ["live_bench/agentic_coding"]:
        # allow running agentic coding benchmark separately from the rest
        session_name += "-agentic-coding"
    
    print(f"Running benchmarks together: {', '.join(params.bench_names)}")
    # Build a single command with all benchmarks to take advantage of parallel execution
    print("Building command with all benchmarks...")
    cmd = build_run_command_from_params(params, bench_name=params.bench_names)
    
    # Set up tmux session
    setup_tmux_session(session_name, [params.bench_names[0]], [cmd], params.venv)

def run_parallel(params: LiveBenchParams) -> None:
    """Run benchmarks in parallel in separate tmux panes"""
    print(f"\nRunning parallel benchmarks for model: {params.model}")
    session_name = f"livebench-{params.model}".replace(".", "_").replace(":", "_")
    
    # If no bench_names provided, use DEFAULT_BENCHMARKS for parallelization
    benchmarks = params.bench_names if params.bench_names else DEFAULT_BENCHMARKS
    print(f"Using benchmarks: {', '.join(benchmarks)}")
    
    # Build commands for each benchmark
    print("Building commands for each benchmark...")
    commands = []
    for bench in benchmarks:
        cmd = build_run_command_from_params(params, bench_name=bench)
        commands.append(cmd)
    
    # Set up tmux session with parallel panes
    setup_tmux_session(session_name, benchmarks, commands, params.venv)

def run_single(params: LiveBenchParams) -> int:
    """
    Run benchmarks together in a single process.
    
    Args:
        params: Parameters for the benchmark run
    
    Returns:
        Exit code from the last command executed
    """
    # If no bench_names provided, use live_bench (runs all benchmarks)
    bench_name = params.bench_names if params.bench_names else "live_bench"
    
    if isinstance(bench_name, list):
        print(f"\nRunning benchmarks together for model: {params.model}")
        print(f"Benchmarks: {', '.join(bench_name)}")
    else:
        print(f"\nRunning benchmark '{bench_name}' for model: {params.model}")
    
    # Build the chained command using build_run_command
    cmd = build_run_command_from_params(params, bench_name=bench_name)
    
    # Run the command
    if params.venv:
        print(f"Activating virtual environment: {params.venv}")
        os.environ["PATH"] = f"{os.path.dirname(params.venv)}:{os.environ['PATH']}"
        run_command(f"source {params.venv}")
    
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
    
    # Get default venv path - check for active venv first
    default_venv = detect_active_venv()
    
    # Required arguments
    parser.add_argument("--model", required=False, default=None, nargs="+", help="One or more model identifiers (e.g., gpt-4)")
    
    # Optional arguments
    parser.add_argument("--venv", help="Path to virtual environment to activate", default=default_venv)
    parser.add_argument("--mode", choices=["single", "parallel", "sequential"], default="single",
                      help="Execution mode: 'single' runs benchmarks together directly (no tmux), "
                           "'sequential' runs benchmarks together in a tmux session, "
                           "'parallel' runs each benchmark in a separate tmux pane. "
                           "Note: 'single' and 'sequential' both collect all questions upfront for shared parallelism.")
    parser.add_argument("--bench-name", nargs="+",
                      help="One or more benchmark paths to run. If not provided: for single/sequential modes, "
                           "defaults to 'live_bench' (all benchmarks together). For parallel mode, uses "
                           "predefined benchmark list with each benchmark in its own pane.")
    parser.add_argument("--question-source", default="huggingface", choices=["huggingface", "jsonl"],
                      help="Source of benchmark questions")
    parser.add_argument("--api-base", help="Base URL for API requests")
    parser.add_argument("--api-key-name", help="Environment variable name containing the API key")
    parser.add_argument("--api-key", help="Direct API key to use for authentication")
    parser.add_argument("--model-display-name", help="Display name for the model in results")
    parser.add_argument("--max-tokens", type=int, help="Maximum tokens for model responses")
    parser.add_argument("--parallel-requests", type=int, help="Number of parallel requests for API calls")
    parser.add_argument("--parallel-grading", type=int, help="Number of parallel grading threads")
    parser.add_argument("--resume", action="store_true", help="Resume from previous run (applies to both inference and grading)")
    parser.add_argument("--resume-inference", action="store_true", help="Resume only for inference (gen_api_answer.py)")
    parser.add_argument("--resume-grading", action="store_true", help="Resume only for grading (gen_ground_truth_judgment.py)")
    parser.add_argument("--retry-failures", action="store_true", help="Retry failed generations")
    parser.add_argument("--skip-inference", action="store_true", help="Skip running gen_api_answer.py")
    parser.add_argument("--skip-grading", action="store_true", help="Skip running gen_ground_truth_judgment.py")
    parser.add_argument("--force-temperature", type=float, help="Force a specific temperature for model sampling")
    parser.add_argument("--num-choices", type=int, help="Number of choices to generate")
    parser.add_argument("--question-begin", type=int, help="Starting question index")
    parser.add_argument("--question-end", type=int, help="Ending question index")
    parser.add_argument("--question-id", nargs="+", help="Specific question IDs to process")
    parser.add_argument("--livebench-release-option", help="LiveBench release option")
    parser.add_argument("--stream", action="store_true", help="Enable streaming mode")
    parser.add_argument("--remove-existing-judgment-file", action="store_true",
                      help="Remove existing judgment file before running")
    parser.add_argument("--only-incorrect", action="store_true",
                      help="When used with --resume-grading or --resume, only re-evaluate questions that previously scored 0")
    parser.add_argument("--ignore-missing-answers", action="store_true",
                      help="Ignore missing answers when running gen_ground_truth_judgment.py")
    parser.add_argument("--debug", action="store_true", 
                      help="Enable debug mode for gen_ground_truth_judgment.py (not passed to gen_api_answer.py)")
    parser.add_argument("--model-provider-override", help="Override the model provider for gen_api_answer.py")
    
    args = parser.parse_args()

    if args.model is None and not args.skip_inference:
        raise ValueError("Model is required when performing inference")
    
    print("\nStarting LiveBench evaluation")
    print(f"Mode: {args.mode}")
    if args.model:
        print(f"Models: {', '.join(args.model)}")
    if args.bench_name:
        print(f"Benchmarks: {', '.join(args.bench_name)}")
    print(f"Question source: {args.question_source}")
    
    # Run each model in its own tmux session
    if args.model:
        for model in args.model:
            # Create params for this model run
            params = LiveBenchParams.from_args(args, model=model)
            run_model(params)
    else:
        params = LiveBenchParams.from_args(args)
        run_model(params)

if __name__ == "__main__":
    main()