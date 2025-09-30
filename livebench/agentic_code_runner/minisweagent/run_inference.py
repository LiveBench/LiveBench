import json
import os
import subprocess
import time

import shortuuid
import yaml

from livebench.model.api_model_config import AgentConfig
from livebench.common import LIVE_BENCH_ROOT_PATH

from livebench.process_results.coding.utils import agentic_coding_process_results
from livebench.model.completions import API_ERROR_OUTPUT


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

# python agentic_code_runner/sweagent/agent/run/run.py run --agent.model.name anthropic/claude-3-5-sonnet-20241022 --env.deployment.image mswebench/ponylang_m_ponyc:pr-4583 --env.repo.path agentic_code_runner/data/repos/ponylang/ponyc --problem_statement.text "make a small change to some file in the repo (just add a meaningless comment somewhere)" --config agentic_code_runner/sweagent/config/livebench.yaml --problem_statement.id test --env_var_path .env
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
    agent_config: AgentConfig | None = None,
    task_to_answer_file: dict[str, str] | None = None
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
    
    run_id = shortuuid.uuid()

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

    config_path = LIVE_BENCH_ROOT_PATH / f"agentic_code_runner/minisweagent/config/livebench.yaml"
    config = yaml.safe_load(open(config_path))

    if provider == 'openai_responses':
        config['model']['api_type'] = 'responses'
        provider = 'openai'
    elif provider == 'google':
        provider = 'gemini'
    elif provider == 'together':
        provider = 'together_ai'

    litellm_info = litellm.model_cost.get(model_api_name, None) or litellm.model_cost.get(provider + '/' + model_api_name, None)
    if litellm_info is None:
        print('Model ' + provider + '/' + model_api_name + ' not registered with litellm')
        if agent_config is None:
            raise ValueError("Model " + model_api_name + " not registered with litellm and not agent configuration provided.")
    
    if config['model'] is None:
        config['model'] = {}

    if config['model']['model_kwargs'] is None:
        config['model']['model_kwargs'] = {}

    if agent_config is not None:
        if 'litellm_provider' in agent_config:
            del agent_config['litellm_provider']
        config['model']['model_kwargs'].update(agent_config)

    config['model']['model_kwargs'].update(api_kwargs)

    orig_api_kwargs = config['model']['model_kwargs'].copy()

    if api_dict is not None and 'http' in provider:
        if api_dict.get('api_base', None) is not None:
            config['model']['model_kwargs']['api_base'] = api_dict['api_base']
            provider = 'openai'

    config['model']['model_name'] = provider + '/' + model_api_name

    if api_dict is not None:
        if api_dict.get('api_key', None) is not None:
            config['model']['model_kwargs']['api_key'] = api_dict['api_key']

    config_path = all_traj_folder / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    if answer_file is not None:
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    
    # Also create directories for task-specific answer files
    if task_to_answer_file is not None:
        for task_answer_file in task_to_answer_file.values():
            os.makedirs(os.path.dirname(task_answer_file), exist_ok=True)

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
            instance_obj = {
                'instance_id': str(question['question_id']),
                'image_name': instance_image_id,
                'problem_statement': question['turns'][0],
                'environment_class': 'docker',
            }
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

    print('Running command: ', ' '.join([str(c) for c in cmd]))

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Stopping subprocess and continuing to collect results...")
        pass

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
        else:
            trajectory = json.load(open(traj_file))

            final_answer = trajectory['info']['submission']
            if final_answer is None:
                final_answer = ""

            del trajectory['info']['submission']

            ans.update({
                'trajectory': json.dumps(trajectory, indent=4),
                'choices': [{'turns': [final_answer]}],
                'total_output_tokens': trajectory['info']['model_stats']['total_output_tokens'],
                'total_input_tokens': trajectory['info']['model_stats']['total_input_tokens'],
            })

    current_answer_file = answer_file
    if task_to_answer_file is not None and 'task' in question:
        task_name = question['task']
        if task_name in task_to_answer_file:
            current_answer_file = task_to_answer_file[task_name]
            # Ensure the directory exists for the task-specific answer file
            os.makedirs(os.path.dirname(current_answer_file), exist_ok=True)

    if current_answer_file is not None:
        with open(current_answer_file, "a") as fout:
            fout.write(json.dumps(ans) + "\n")