from dataclasses import dataclass
import yaml
import os
from functools import cache

APIKwargs = dict[str, dict[str, str | int | float | bool | dict[str, str] | None]]

@dataclass
class ModelConfig:
    display_name: str # Name of the model as it will be displayed in the leaderboard; used for file naming
    api_name: dict[str, str] # mapping of provider name to model name on that provider's API
    default_provider: str | None = None # provider to use if not otherwise specified
    api_keys: dict[str, str] | None = None # mapping of provider name to API key
    aliases: list[str] | None = None # alternative names for the model
    api_kwargs: APIKwargs | None = None # mapping of provider name to additional arguments to pass to the API call
    prompt_prefix: str | None = None # prefix to add to the prompt
    preserve_reasoning: bool | None = None # whether to preserve reasoning in multi-turn tasks

@cache
def load_model_configs(file_path: str) -> dict[str, ModelConfig]:
    with open(file_path, 'r') as file:
        model_configs_list = yaml.safe_load_all(file)

        model_configs: dict[str, ModelConfig] = {}
        for model_config in model_configs_list:
            if model_config is not None:
                model_configs[model_config['display_name']] = ModelConfig(**model_config)

    return model_configs

MODEL_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), 'model_configs')
@cache
def load_all_configs() -> dict[str, ModelConfig]:
    model_configs: dict[str, ModelConfig] = {}
    for file in os.listdir(MODEL_CONFIGS_DIR):
        model_configs.update(load_model_configs(os.path.join(MODEL_CONFIGS_DIR, file)))
    return model_configs

@cache
def get_model_config(model_name: str) -> ModelConfig:
    model_configs = load_all_configs()
    matches: list[ModelConfig] = []
    for model_config in model_configs.values():
        if model_name.lower() == model_config.display_name.lower() or (model_config.aliases and model_name.lower() in [alias.lower() for alias in model_config.aliases]):
            matches.append(model_config)
    if len(matches) > 1:
        raise ValueError(f"Multiple model configs found for {model_name}")
    elif len(matches) == 0:
        return ModelConfig(
            display_name=model_name.lower(),
            api_name={
                'local': model_name
            },
            aliases=[],
            api_kwargs={}
        )
    else:
        return matches[0]
