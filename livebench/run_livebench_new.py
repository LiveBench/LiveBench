import argparse
import os
from livebench.common import LIVE_BENCH_RELEASES, filter_questions_for_inference, load_questions
from livebench.model.api_model_config import ModelConfig, prepare_model_config
from livebench.gen_api_answer import run_inference
from collections import defaultdict



def main():
    parser = argparse.ArgumentParser(description="Run LiveBench benchmarks with various execution modes")
    
    parser.add_argument("--model", required=False, default=[], nargs="+", help="One or more model identifiers (e.g., gpt-4)")
    
    parser.add_argument("--bench-name", nargs="+", default=["live_bench"],
                      help="One or more benchmark paths to run. If not provided: for single/sequential modes, "
                           "runs all benchmarks in sequence using 'live_bench'. For parallel mode, uses "
                           "predefined benchmark list for parallelization.")
    parser.add_argument("--question-source", default="huggingface", choices=["huggingface", "jsonl"],
                      help="Source of benchmark questions")
    parser.add_argument("--api-base", help="Base URL for API requests")
    parser.add_argument("--api-key-name", help="Environment variable name containing the API key")
    parser.add_argument("--api-key", help="Direct API key to use for authentication")
    parser.add_argument("--force-model-display-name", help="Display name for the model")
    parser.add_argument("--force-model-provider", help="Provider name for the model")
    parser.add_argument("--force-max-tokens", type=int, help="Maximum tokens for model responses")
    parser.add_argument("--parallel-requests", type=int, default=1, help="Number of parallel requests for API calls")
    parser.add_argument("--parallel-grading", type=int, default=1, help="Number of parallel grading threads")
    parser.add_argument("--resume", action="store_true", help="Resume from previous run (applies to both inference and grading)")
    parser.add_argument("--resume-inference", action="store_true", help="Resume only for inference (gen_api_answer.py)")
    parser.add_argument("--resume-grading", action="store_true", help="Resume only for grading (gen_ground_truth_judgment.py)")
    parser.add_argument("--retry-failures", action="store_true", help="Retry failed generations")
    parser.add_argument("--skip-inference", action="store_true", help="Skip running gen_api_answer.py")
    parser.add_argument("--skip-grading", action="store_true", help="Skip running gen_ground_truth_judgment.py")
    parser.add_argument("--force-temperature", type=float, help="Force a specific temperature for model sampling")
    parser.add_argument("--num-choices", type=int, help="Number of choices to generate", default=1)
    parser.add_argument("--question-id", nargs="+", help="Specific question IDs to process")
    parser.add_argument("--livebench-release-option", help="LiveBench release option", default=max(LIVE_BENCH_RELEASES))
    parser.add_argument("--stream", action="store_true", help="Enable streaming mode")
    parser.add_argument("--remove-existing-judgments", action="store_true", 
                      help="Remove existing judgment file before running")
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

    assert len(set(bench.split('/')[0] for bench in args.bench_name)) == 1, "All benchmarks must have the same root name"

    bench_name_root = args.bench_name[0].split('/')[0]

    all_questions = load_questions(
        question_source=args.question_source,
        bench_name=args.bench_name,
        livebench_release_option=args.livebench_release_option,
        question_ids=args.question_id
    )

    api_dict = {}
    if args.api_key:
        api_dict['api_key'] = args.api_key
    elif args.api_key_name:
        api_dict['api_key'] = os.environ[args.api_key_name]
    if args.api_base:
        api_dict['api_base'] = args.api_base
        if 'api_key' not in api_dict:
            api_dict['api_key'] = os.environ.get("LIVEBENCH_API_KEY", "EMPTY")

    # model name -> category -> task -> list of question ids
    questions_by_model = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    model_configs: dict[str, ModelConfig] = {}
    for model in args.model:
        model_config = prepare_model_config(
            model_name=model, 
            api_dict=api_dict, 
            force_max_tokens=args.force_max_tokens, 
            force_temperature=args.force_temperature,
            model_provider_override=args.force_model_provider,
            model_display_name_override=args.force_model_display_name,
        )
        model_configs[model_config.display_name] = model_config
        for category in all_questions:
            for task in all_questions[category]:
                bench_name = bench_name_root + '/' + category + '/' + task
                questions_by_model[model_config.display_name][category][task] = filter_questions_for_inference(all_questions[category][task], bench_name, model_config.display_name, args.resume, args.retry_failures)

    # Run inference if not skipped
    if not args.skip_inference:
        
        print(f"\nRunning inference with {args.parallel_requests} parallel requests per model")
        
        run_inference(
            all_questions=all_questions,
            questions_by_model=questions_by_model,
            model_configs=model_configs,
            parallel_requests=args.parallel_requests,
            num_choices=args.num_choices,
            stream=args.stream,
            bench_name_root=bench_name_root,
        )
        
        print("Inference completed successfully")
    else:
        print("Skipping inference")

if __name__ == '__main__':
    main()