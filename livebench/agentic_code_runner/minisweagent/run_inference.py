import json
import os
import threading
import time
import subprocess
import shortuuid
import yaml

from livebench.common import LIVE_BENCH_ROOT_PATH

from livebench.process_results.coding.utils import agentic_coding_process_results
from livebench.model.completions import API_ERROR_OUTPUT
from livebench.model.api_model_config import get_model_config, RESPONSES_API_PROVIDERS


def update_dict_recursively(d1, d2):
    """
    Recursively update dict d1 with dict d2
    """
    for k, v in d2.items():
        if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
            update_dict_recursively(d1[k], v)
        else:
            d1[k] = v
    return d1



RUN_SUCCESS_EXIT_STATUS = "Submitted"


def _eval_status_from_exit_status(exit_status):
    if not exit_status or exit_status == RUN_SUCCESS_EXIT_STATUS:
        return None
    if exit_status == "LimitsExceeded":
        return "run_limits_exceeded"
    return "run_error"

def _incremental_grading_loop(
    questions: list[dict],
    model_name: str,
    all_traj_folder,
    grading_parallel: int,
    stop_event: threading.Event,
    poll_seconds: int = 20,
):
    """Grader pool: grade agentic instances as they land instead of waiting for the
    whole answer round. Polls the trajectory folder for newly written
    <qid>.traj.json files and feeds each batch of completions through
    agentic_coding_process_results, whose eval cache records the scores; the final
    judgment pass then reuses them (hash-matched on the exact patch) instead of
    re-running the docker evals. Failures here are safe: anything ungraded or
    uncached is simply graded by the final pass as before.
    """
    questions_by_qid = {str(q['question_id']): q for q in questions}
    handled: set[str] = set()
    while True:
        stopping = stop_event.is_set()
        ready_questions, ready_answers = [], []
        for qid, question in questions_by_qid.items():
            if qid in handled:
                continue
            traj_file = all_traj_folder / qid / f"{qid}.traj.json"
            if not traj_file.exists():
                continue
            try:
                with open(traj_file) as f:
                    trajectory = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue  # mid-write; the next poll picks it up
            info = trajectory.get('info', {})
            if _eval_status_from_exit_status(info.get('exit_status')) == 'run_error':
                # infra failure — the collection pass writes $ERROR$ for it and
                # --retry-failures re-runs it; nothing gradable here
                handled.add(qid)
                continue
            # Same string the collection pass will put in choices[0].turns[0],
            # so the cache's patch hash matches at judgment time.
            submission = info.get('submission')
            if submission is None:
                submission = ""
            ready_questions.append(question)
            ready_answers.append({
                'question_id': question['question_id'],
                'model_id': model_name,
                'choices': [{'turns': [submission]}],
            })
        if ready_questions:
            print(f"incremental grader: grading {len(ready_questions)} completed instances "
                  f"({len(handled) + len(ready_questions)}/{len(questions_by_qid)} handled)")
            try:
                agentic_coding_process_results(
                    ready_questions, ready_answers, debug=False, max_workers=grading_parallel)
            except Exception as e:
                print(f"incremental grader: batch failed ({e}); the final grading pass will cover it")
            handled.update(str(q['question_id']) for q in ready_questions)
        if stopping:
            break
        stop_event.wait(timeout=poll_seconds)
    print(f"incremental grader: done ({len(handled)}/{len(questions_by_qid)} instances handled)")


def run_agentic_coding_inference(
    questions: list[dict],
    model_api_name: str,
    provider: str,
    force_temperature: float | None,
    num_choices: int,
    model_api_kwargs: dict[str, str] | None = None,
    api_dict: dict[str, str] | None = None,
    model_display_name_override: str | None = None,
    answer_file: str | None = None,
    parallel: int = 1,
    replay_traj_dir: str | None = None,
    custom_run_id: str | None = None,
    preserve_reasoning: bool | None = None,
    grading_parallel: int = 0,
):

    if len(questions) == 0:
        return

    import litellm
    from livebench.agentic_code_runner.eval.utils import docker_util
    if force_temperature is not None:
        temperature = force_temperature
    else:
        temperature = 0

    if num_choices != 1:
        raise ValueError("num_choices must be 1 for agentic coding")
    
    run_id = custom_run_id if custom_run_id else shortuuid.uuid()

    model_name = model_display_name_override if model_display_name_override else model_api_name

    api_kwargs = {
        'temperature': temperature
    }

    if 'max_tokens' in api_kwargs:
        del api_kwargs['max_tokens']

    if 'max_completion_tokens' in api_kwargs:
        del api_kwargs['max_completion_tokens']

    if model_api_kwargs is not None:
        model_api_kwargs = {key: value for key, value in model_api_kwargs.items()}
        api_kwargs.update(model_api_kwargs)

    all_traj_folder = LIVE_BENCH_ROOT_PATH / f"agentic_code_runner/data/trajectories" / run_id
    all_traj_folder.mkdir(parents=True, exist_ok=True)

    # Native tool calling needs the tool-calling prompts: under tool_choice='auto' the
    # triple-backtick instructions in livebench.yaml compete with the tool and models
    # ignore it. Anthropic always runs native. ALSO required for endpoints that REJECT a
    # forced tool_choice while thinking is on — DashScope/qwen and DeepSeek's own API both
    # 400 with "does not support being set to required ... in thinking mode" — because
    # there the prompt template is the only lever left. Measured on qwen3.8-max under the
    # text prompts: it called the bash tool on just 19/72 questions.
    # Match on the api_base too, not just the provider string: a model reached through an
    # explicit base URL arrives here as provider='openai_responses'/'openai', so a
    # provider-only check silently misses it. Meta (api.meta.ai, muse-spark) is exactly
    # that case -- it 400s on a forced tool_choice, so it needs 'auto', and 'auto' WITHOUT
    # the native prompts is the qwen failure mode: the triple-backtick instructions win and
    # the model ignores the tool (19/72 questions on qwen3.8-max).
    _NO_FORCED_TOOL_CHOICE_HOSTS = ('dashscope', 'aliyuncs', 'deepseek', 'meta.ai')
    _endpoint = f"{provider} {(api_dict or {}).get('api_base', '')}".lower()
    _no_forced_tool_choice = any(h in _endpoint for h in _NO_FORCED_TOOL_CHOICE_HOSTS)
    native_tools = provider == 'anthropic' or _no_forced_tool_choice
    config_name = "livebench_native.yaml" if native_tools else "livebench.yaml"
    if native_tools:
        print(f"Native tool-calling mode ON: using {config_name}")
    config_path = LIVE_BENCH_ROOT_PATH / f"agentic_code_runner/minisweagent/config/{config_name}"
    config = yaml.safe_load(open(config_path))

    if provider in RESPONSES_API_PROVIDERS:
        config['model']['api_type'] = 'responses'
        provider = RESPONSES_API_PROVIDERS[provider]
    elif provider == 'google':
        provider = 'gemini'
    elif provider == 'together':
        provider = 'together_ai'

    litellm_info = litellm.model_cost.get(model_api_name, None) or litellm.model_cost.get(provider + '/' + model_api_name, None)
    if litellm_info is None:
        print('Model ' + provider + '/' + model_api_name + ' not registered with litellm')
    
    if config['model'] is None:
        config['model'] = {}

    if config['model']['model_kwargs'] is None:
        config['model']['model_kwargs'] = {}

    config['model']['model_kwargs'].update(api_kwargs)

    orig_api_kwargs = config['model']['model_kwargs'].copy()

    if api_dict is not None:
        if api_dict.get('api_base', None) is not None:
            config['model']['model_kwargs']['api_base'] = api_dict['api_base']
            provider = 'openai'
        if api_dict.get('api_key', None) is not None:
            config['model']['model_kwargs']['api_key'] = api_dict['api_key']

    config['model']['model_name'] = provider + '/' + model_api_name

    if preserve_reasoning:
        config['model']['preserve_reasoning'] = True

    # Explicit native-tools override from the model config (agentic_native_tools key).
    # Without it the channel is inferred from name substrings, which silently splits
    # paired legs when a served alias lacks the family token (2026-08-30).
    try:
        _nt = getattr(get_model_config(model_name), 'agentic_native_tools', None)
    except Exception:
        _nt = None
    if _nt is not None:
        config['model']['native_tools'] = _nt

    config_path = all_traj_folder / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    if answer_file is not None:
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    
    # Create directories for question-specific answer files
    for question in questions:
        if 'answer_file' in question and answer_file is None:
            os.makedirs(os.path.dirname(question['answer_file']), exist_ok=True)

    for question in questions:
        instance_image_id = f"mswebench/{question['org']}_m_{question['repo']}:pr-{question['number']}"
        if not docker_util.exists(instance_image_id):
            # run eval harness to build image
            answers = [{'question_id': question['question_id'], 'choices': [{'turns': ['placeholder']}], 'model_id': 'image_build'} for question in questions]
            print(f"Building image for {instance_image_id}")
            agentic_coding_process_results(questions, answers, debug=False, max_workers=parallel, only_build_image=True)

        problem_statement_text = question['turns'][0]
        problem_statement_path = LIVE_BENCH_ROOT_PATH / f'agentic_code_runner/data/problem_statements/{question["question_id"]}'
        problem_statement_path.parent.mkdir(parents=True, exist_ok=True)
        with open(problem_statement_path, 'w') as f:
            f.write(problem_statement_text)

        traj_folder = all_traj_folder / str(question['question_id'])
        traj_folder.mkdir(parents=True, exist_ok=True)

    instances_path = LIVE_BENCH_ROOT_PATH / f'agentic_code_runner/data/instances/{model_name}.jsonl'
    instances_path.parent.mkdir(parents=True, exist_ok=True)
    with open(instances_path, 'w') as f:
        for question in questions:
            if (all_traj_folder / str(question['question_id'])).exists() and f"{question['question_id']}.pred" in os.listdir(all_traj_folder / str(question['question_id'])):
                print(f"Skipping {question['question_id']} because it already exists")
                continue
            instance_image_id = f"mswebench/{question['org']}_m_{question['repo']}:pr-{question['number']}"
            if not question.get('task', None):
                raise ValueError("Task is required for minisweagent")
            if not question.get('repo', None):
                raise ValueError("Repo is required for minisweagent")
            instance_obj = {
                'instance_id': str(question['question_id']),
                'image_name': instance_image_id,
                'problem_statement': question['turns'][0],
                'environment_class': 'docker',
            }
            if question['task'] != 'python':
                instance_obj['cwd'] = '/home/' + question['repo']
            f.write(json.dumps(instance_obj) + '\n')
    
    run_script = LIVE_BENCH_ROOT_PATH / 'agentic_code_runner/minisweagent/run/run_single.py' if parallel == 1 else LIVE_BENCH_ROOT_PATH / 'agentic_code_runner/minisweagent/run/run_batch.py'
    cmd = [
        'python',
        run_script,
        '--instances_path',
        instances_path,
        '--config',
        config_path,
        '--output',
        all_traj_folder,
    ]
    if parallel > 1:
        cmd.extend(['--workers', str(parallel)])
    
    if replay_traj_dir is not None:
        if parallel == 1:
            cmd.extend(['--replay-traj', replay_traj_dir])
        else:
            cmd.extend(['--replay-traj-dir', replay_traj_dir])

    print('Running command: ', ' '.join([str(c) for c in cmd]))

    # Grader pool: grade instances as their trajectories land instead of leaving
    # all docker evals to the judgment phase. Replay runs are excluded (their
    # trajectories pre-exist, so everything would grade before the replay ran).
    grader_stop = None
    grader_thread = None
    if grading_parallel > 0 and replay_traj_dir is None:
        print(f"Incremental grading ON: grader pool of {grading_parallel} alongside the answer phase")
        grader_stop = threading.Event()
        grader_thread = threading.Thread(
            target=_incremental_grading_loop,
            args=(questions, model_name, all_traj_folder, grading_parallel, grader_stop),
            daemon=True,
        )
        grader_thread.start()

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Stopping subprocess and continuing to collect results...")
        pass
    except subprocess.CalledProcessError as e:
        print(f"Subprocess error: {e}")
        pass
    finally:
        if grader_thread is not None:
            grader_stop.set()
            grader_thread.join()

    for question in questions:

        ans = {
            'question_id': question['question_id'],
            'answer_id': shortuuid.uuid(),
            'run_id': run_id,
            'model_id': model_name,
            'tstamp': time.time(),
            'api_info': {
                'provider': api_dict['api_base'] if api_dict and 'api_base' in api_dict else provider,
                'api_name': model_api_name,
                'api_kwargs': orig_api_kwargs
            }
        }

        traj_folder = all_traj_folder / str(question['question_id'])
        traj_file = traj_folder / f"{question['question_id']}.traj.json"

        if not traj_file.exists():
            print(f"Trajectory file {traj_file} does not exist")
            ans['choices'] = [{'turns': [API_ERROR_OUTPUT]}]
            ans['eval_status'] = "run_no_trajectory"
        else:
            trajectory = json.load(open(traj_file))

            exit_status = trajectory['info'].get('exit_status')
            eval_status = _eval_status_from_exit_status(exit_status)
            if eval_status is not None:
                ans['eval_status'] = eval_status
                print(f"Run for question {question['question_id']} ended with exit_status={exit_status} -> eval_status={eval_status}")

            final_answer = trajectory['info']['submission']
            if final_answer is None:
                final_answer = ""

            if eval_status == 'run_error':
                # Infra failure (e.g. docker exec timeout) — the submission holds the
                # error text, not a patch. Write $ERROR$ so --retry-failures re-runs it.
                ans['error'] = exit_status
                ans['error_msg'] = final_answer
                final_answer = API_ERROR_OUTPUT

            del trajectory['info']['submission']

            stats = trajectory['info'].get('model_stats', {}) or {}
            in_tok = stats.get('total_input_tokens') or 0
            out_tok = stats.get('total_output_tokens') or 0
            cached_tok = stats.get('total_cached_tokens') or 0
            cache_creation_tok = stats.get('total_cache_creation_tokens') or 0
            cost_usd = stats.get('instance_cost')
            try:
                cpm = get_model_config(model_name).cost_per_million
            except Exception:
                cpm = None
            if cpm:
                cached = cached_tok if 'cached_input' in cpm else 0
                uncached = max(in_tok - cached, 0)
                cost_usd = round(
                    (uncached / 1_000_000) * cpm.get('input', 0)
                    + (cached / 1_000_000) * cpm.get('cached_input', cpm.get('input', 0))
                    + (out_tok / 1_000_000) * cpm.get('output', 0),
                    6,
                )

            ans.update({
                'trajectory': json.dumps(trajectory, indent=4),
                'choices': [{'turns': [final_answer]}],
                'total_output_tokens': out_tok,
                'total_input_tokens': in_tok,
                'total_cached_tokens': cached_tok,
                'total_cache_creation_tokens': cache_creation_tok,
                'n_model_calls': stats.get('api_calls'),
                # surfaced top-level so consumers (finalize.sh's native_tools_effective,
                # health scans) don't have to parse the embedded trajectory JSON
                'native_tool_use_turns': stats.get('native_tool_use_turns'),
                'total_time_s': stats.get('run_time_s'),
                'model_cost': stats.get('instance_cost'),
                'cost_usd': cost_usd,
            })

        # Use answer_file parameter if provided (as override), otherwise use question's answer_file
        current_answer_file = answer_file if answer_file is not None else question.get('answer_file')
        
        if current_answer_file is not None:
            with open(current_answer_file, "a") as fout:
                fout.write(json.dumps(ans) + "\n")