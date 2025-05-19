import json
import os
import subprocess
import time

import shortuuid
import yaml

from livebench.model.api_model_config import AgentConfig
from livebench.common import LIVE_BENCH_ROOT_PATH
from livebench.agentic_code_runner.eval.utils import docker_util
from livebench.process_results.coding.utils import agentic_coding_process_results
from livebench.model.completions import API_ERROR_OUTPUT

from collections import defaultdict

import litellm

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
    agent_config: AgentConfig | None = None
):
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

    if model_api_kwargs is not None:
        model_api_kwargs = {key: value for key, value in model_api_kwargs.items()}
        api_kwargs.update(model_api_kwargs)

    if 'max_tokens' in api_kwargs:
        del api_kwargs['max_tokens']

    if 'max_completion_tokens' in api_kwargs:
        del api_kwargs['max_completion_tokens']

    orig_api_kwargs = api_kwargs.copy()

    all_traj_folder = LIVE_BENCH_ROOT_PATH / f"agentic_code_runner/data/trajectories" / run_id
    all_traj_folder.mkdir(parents=True, exist_ok=True)

    base_config_path = LIVE_BENCH_ROOT_PATH / 'agentic_code_runner/sweagent/config/livebench_base.yaml'
    function_calling_config_path = LIVE_BENCH_ROOT_PATH / 'agentic_code_runner/sweagent/config/function_calling.yaml'
    thought_action_config_path = LIVE_BENCH_ROOT_PATH / 'agentic_code_runner/sweagent/config/thought_action.yaml'
    config = yaml.safe_load(open(base_config_path))

    if provider == 'openai_responses':
        config['agent']['model']['api_type'] = 'responses'
        provider = 'openai'
    elif provider == 'google':
        provider = 'gemini'
    elif provider == 'together':
        provider = 'together_ai'

    litellm_info = litellm.model_cost.get(model_api_name, None) or litellm.model_cost.get(provider + '/' + model_api_name, None)
    if litellm_info is None:
        print('Model ' + provider + '/' + model_api_name + ' not registered with litellm')
        if agent_config is not None:
            raise ValueError("Model " + model_api_name + " not registered with litellm and not agent configuration provided.")


    if agent_config is not None and 'supports_function_calling' in agent_config:
        if agent_config['supports_function_calling']:
            use_function_calling = True
        else:
            use_function_calling = False
        del agent_config['supports_function_calling']
    else:
        if (litellm.utils.supports_function_calling(model=model_api_name) or litellm.utils.supports_function_calling(model=provider + '/' + model_api_name)):
            use_function_calling = True
        else:
            use_function_calling = False

    if use_function_calling:
        function_calling_config = yaml.safe_load(open(function_calling_config_path))
        update_dict_recursively(config, function_calling_config)
        print("Using function calling config")
    else:
        thought_action_config = yaml.safe_load(open(thought_action_config_path))
        update_dict_recursively(config, thought_action_config)
        print("Using thought action config")

    # swe-agent has temperature and top p as top-level config keys so we set them here
    # and don't pass them as extra completion kwargs
    if 'temperature' in api_kwargs:
        if api_kwargs['temperature'] is not None:
            config['agent']['model']['temperature'] = api_kwargs['temperature']
        else:
            config['agent']['model']['temperature'] = 1
        del api_kwargs['temperature']

    if 'top_p' in api_kwargs:
        if api_kwargs['top_p'] is not None:
            config['agent']['model']['top_p'] = api_kwargs['top_p']
        else:
            config['agent']['model']['top_p'] = None
        del api_kwargs['top_p']

    if agent_config is not None:
        if 'litellm_provider' in agent_config:
            del agent_config['litellm_provider']
        config['agent']['model'].update(agent_config)

    config['agent']['model']['completion_kwargs'] = api_kwargs

    config_path = all_traj_folder / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    agent_run_path = LIVE_BENCH_ROOT_PATH / 'agentic_code_runner/sweagent/agent/run/run.py'

    if answer_file is not None:
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)

    env_var_path = LIVE_BENCH_ROOT_PATH / '.env'

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
        # if traj_folder.exists():
        #     shutil.rmtree(traj_folder)
        traj_folder.mkdir(parents=True, exist_ok=True)

    if parallel == 1:
        for question in questions:
            if (all_traj_folder / str(question['question_id'])).exists() and f"{question['question_id']}.pred" in os.listdir(all_traj_folder / str(question['question_id'])):
                print(f"Skipping {question['question_id']} because it already exists")
                continue
            instance_image_id = f"mswebench/{question['org']}_m_{question['repo']}:pr-{question['number']}"
            problem_statement_path = LIVE_BENCH_ROOT_PATH / f'agentic_code_runner/data/problem_statements/{question["question_id"]}'
            repo = question['repo']
            org = question['org']
            repo_path = LIVE_BENCH_ROOT_PATH / f'agentic_code_runner/data/repos/{org}/{repo}'
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
                str(question['question_id']),
                '--env_var_path',
                env_var_path,
                '--config',
                config_path,
                '--env.deployment.type',
                'docker',
                '--env.deployment.python_standalone_dir',
                'None',
                '--env.repo.type',
                'local',
                '--output_dir',
                all_traj_folder
            ]

            print('Running command: ', ' '.join([str(c) for c in cmd]))

            subprocess.run(cmd, check=True)
    elif len(questions) > 0:
        instances_path = LIVE_BENCH_ROOT_PATH / f'agentic_code_runner/data/instances/{model_name}.jsonl'
        instances_path.parent.mkdir(parents=True, exist_ok=True)
        with open(instances_path, 'w') as f:
            for question in questions:
                instance_image_id = f"mswebench/{question['org']}_m_{question['repo']}:pr-{question['number']}"
                instance_obj = {
                    'instance_id': str(question['question_id']),
                    'image_name': instance_image_id,
                    'problem_statement': question['turns'][0],
                    'repo_name': question['repo'],
                    'env': {
                        'deployment': {
                            'type': 'docker',
                            'python_standalone_dir': None
                        },
                    }
                }
                f.write(json.dumps(instance_obj) + '\n')

        cmd = [
            'python',
            agent_run_path,
            'run-batch',
            '--agent.model.name',
            provider + '/' + model_api_name,
            '--instances.type',
            'file',
            '--instances.path',
            instances_path,
            '--config',
            config_path,
            '--env_var_path',
            env_var_path,
            '--num_workers',
            str(parallel),
            '--output_dir',
            all_traj_folder
        ]

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
                'provider': provider if provider != 'local' else api_dict['api_base'],
                'api_name': model_api_name,
                'api_kwargs': orig_api_kwargs
            }
        }

        trajectories = defaultdict(dict)
        preds = defaultdict(dict)

        traj_folder = all_traj_folder / str(question['question_id'])

        pred_file = traj_folder / f"{question['question_id']}.pred"
        traj_file = traj_folder / f"{question['question_id']}.traj"

        if not pred_file.exists() or not traj_file.exists():
            print(f"Prediction file {pred_file} or trajectory file {traj_file} does not exist")
            ans['choices'] = [{'turns': [API_ERROR_OUTPUT]}]
        else:
            trajectory = json.load(open(traj_file))
            pred = json.load(open(pred_file))

            trajectories[model_name][question['question_id']] = trajectory
            preds[model_name][question['question_id']] = pred

            final_answer = preds[model_name][question['question_id']]['model_patch']
            if final_answer is None:
                final_answer = ""

            run_info = trajectories[model_name][question['question_id']]
            history = json.dumps(run_info['history'], indent=4)

            total_output_tokens = run_info['info']['model_stats']['tokens_received']
            total_input_tokens = run_info['info']['model_stats']['tokens_sent']
            cost = run_info['info']['model_stats']['instance_cost']
            api_calls = run_info['info']['model_stats']['api_calls']
            exit_status = run_info['info']['exit_status']
            
            ans.update({
                'total_output_tokens': total_output_tokens,
                'total_input_tokens': total_input_tokens,
                'cost': cost,
                'api_calls': api_calls,
                'history': history,
                'exit_status': exit_status,
                'choices': [{'turns': [final_answer]}]
            })

        if answer_file is not None:
            with open(answer_file, "a") as fout:
                fout.write(json.dumps(ans) + "\n")
