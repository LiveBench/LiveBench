from dataclasses import dataclass
import yaml
import os
from functools import cache

@dataclass
class ModelConfig:
    display_name: str
    api_name: dict[str, str]
    default_provider: str | None = None
    api_keys: dict[str, str] | None = None
    aliases: list[str] | None = None
    api_kwargs: dict[str, dict[str, str | int | float | bool | dict[str, str] | None]] | None = None
    prompt_prefix: str | None = None

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
        if model_name == model_config.display_name or (model_config.aliases and model_name in model_config.aliases):
            matches.append(model_config)
    if len(matches) > 1:
        raise ValueError(f"Multiple model configs found for {model_name}")
    elif len(matches) == 0:
        return ModelConfig(
            display_name=model_name,
            api_name={
                'local': model_name
            },
            aliases=[],
            api_kwargs={}
        )
    else:
        return matches[0]
