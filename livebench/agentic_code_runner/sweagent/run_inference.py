import json
import os
import subprocess
import time

import shortuuid

from livebench.common import LIVE_BENCH_ROOT_PATH

# python agentic_code_runner/sweagent/agent/run/run.py run --agent.model.name anthropic/claude-3-5-sonnet-20241022 --env.deployment.image mswebench/ponylang_m_ponyc:pr-4583 --env.repo.path agentic_code_runner/data/repos/ponylang/ponyc --problem_statement.text "make a small change to some file in the repo (just add a meaningless comment somewhere)" --config agentic_code_runner/sweagent/config/livebench.yaml --problem_statement.id test --env_var_path .env
def run_agentic_coding_inference(
    questions: list[dict],
    model_api_name: str,
    provider: str,
    force_temperature: float | None,
    num_choices: int,
    api_kwargs: dict[str, str] | None = None,
    api_dict: dict[str, str] | None = None,
    stream: bool = False,
    model_display_name_override: str | None = None,
    answer_file: str | None = None
):
    if force_temperature is not None:
        temperature = force_temperature
    else:
        temperature = 0

    if num_choices != 1:
        raise ValueError("num_choices must be 1 for agentic coding")

    agent_config_path = LIVE_BENCH_ROOT_PATH / 'agentic_code_runner/sweagent/config/livebench.yaml'
    agent_run_path = LIVE_BENCH_ROOT_PATH / 'agentic_code_runner/sweagent/agent/run/run.py'

    all_traj_folder = LIVE_BENCH_ROOT_PATH / f"agentic_code_runner/data/trajectories/"
    all_traj_folder.mkdir(parents=True, exist_ok=True)

    if answer_file is not None:
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)

    for question in questions:
        instance_id = question['instance_id']
        instance_image_id = f"mswebench/{instance_id.replace('__', '_m_').replace('-', ':pr-')}"
        repo = question['repo']
        org = question['org']
        repo_path = LIVE_BENCH_ROOT_PATH / f'agentic_code_runner/data/repos/{org}/{repo}'
        problem_statement_text = question['turns'][0]
        problem_statement_path = LIVE_BENCH_ROOT_PATH / f'agentic_code_runner/data/problem_statements/{question["question_id"]}'
        problem_statement_path.parent.mkdir(parents=True, exist_ok=True)
        with open(problem_statement_path, 'w') as f:
            f.write(problem_statement_text)
        problem_statement_id = question['question_id']

        env_var_path = LIVE_BENCH_ROOT_PATH / '.env'

        cmd = [
            'python',
            agent_run_path,
            'run',
            '--agent.model.name',
            provider + '/' + model_api_name,
            '--env.deployment.image',
            instance_image_id,
            '--env.repo.path',
            repo_path,
            '--problem_statement.path',
            problem_statement_path,
            '--problem_statement.id',
            str(problem_statement_id),
            '--env_var_path',
            env_var_path,
            '--config',
            agent_config_path,
            '--agent.model.temperature',
            str(temperature),
        ]

        print('Running command: ', ' '.join(cmd))

        subprocess.run(cmd, check=True)

        traj_folder = all_traj_folder / f"{agent_config_path.stem}__{provider}_{model_api_name}__{problem_statement_id}/{problem_statement_id}"

        pred_file = traj_folder / f"{problem_statement_id}.pred"
        traj_file = traj_folder / f"{problem_statement_id}.traj"

        if not pred_file.exists():
            raise FileNotFoundError(f"Prediction file {pred_file} does not exist")
        
        if not traj_file.exists():
            raise FileNotFoundError(f"Trajectory file {traj_file} does not exist")

        final_answer = json.load(open(pred_file))['model_patch']
        if final_answer is None:
            final_answer = ""

        run_info = json.load(open(traj_file))
        history = json.dumps(run_info['history'], indent=4)

        total_output_tokens = run_info['model_stats']['tokens_received']
        total_input_tokens = run_info['model_stats']['tokens_sent']
        cost = run_info['model_stats']['cost']
        api_calls = run_info['model_stats']['api_calls']
        
        ans = {
            'question_id': problem_statement_id,
            'answer_id': shortuuid.uuid(),
            'model_id': model_display_name_override if model_display_name_override else model_api_name,
            'choices': [final_answer],
            'tstamp': time.time(),
            'total_output_tokens': total_output_tokens,
            'total_input_tokens': total_input_tokens,
            'cost': cost,
            'api_calls': api_calls,
            'history': history,
            'api_info': {
                'provider': provider if provider != 'local' else api_dict['api_base'],
                'api_name': model_api_name,
                'api_kwargs': api_kwargs
            }
        }

        if answer_file is not None:
            with open(answer_file, "a") as fout:
                fout.write(json.dumps(ans) + "\n")
