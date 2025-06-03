from dataclasses import dataclass, field
from typing import Literal
import yaml
import os
from functools import cache
from livebench.common import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE

from typing import TypedDict, Any
try:
    from typing import NotRequired
except:
    from typing_extensions import NotRequired

class AgentConfig(TypedDict):
    litellm_provider: NotRequired[str]
    max_input_tokens: NotRequired[int]
    max_output_tokens: NotRequired[int]
    supports_function_calling: NotRequired[bool]
    api_type: NotRequired[Literal["completion", "responses"]]
    convert_system_to_user: NotRequired[bool]
    include_thinking_in_history: NotRequired[bool]

@dataclass
class FullModelConfig:
    display_name: str # Name of the model as it will be displayed in the leaderboard; used for file naming
    api_name: dict[str, str] # mapping of provider name to model name on that provider's API
    default_provider: str | None = None # provider to use if not otherwise specified
    api_keys: dict[str, str] | None = None # mapping of provider name to API key
    aliases: list[str] | None = None # alternative names for the model
    api_kwargs: dict[str, dict[str, str | int | float | bool | dict[str, str] | None]] = {} # mapping of provider name to additional arguments to pass to the API call
    prompt_prefix: str | None = None # prefix to add to the prompt
    agent_config: dict[str, AgentConfig] | None = None # mapping of provider name to additional configuration for use in agentic coding

    def __post_init__(self):
        if self.agent_config is not None:
            agent_config = {}
            for provider, config in self.agent_config.items():
                agent_config[provider] = AgentConfig(**config)
            self.agent_config = agent_config

@dataclass
class ModelConfig:
    display_name: str
    api_name: str
    api_provider: str
    api_dict: dict[str, str] | None = None
    api_kwargs: dict[str, Any] = field(default_factory=dict)
    prompt_prefix: str | None = None
    agent_config: AgentConfig = field(default_factory=AgentConfig)

@cache
def load_model_configs(file_path: str) -> dict[str, FullModelConfig]:
    with open(file_path, 'r') as file:
        model_configs_list = yaml.safe_load_all(file)

        model_configs: dict[str, FullModelConfig] = {}
        for model_config in model_configs_list:
            if model_config is not None:
                model_configs[model_config['display_name']] = FullModelConfig(**model_config)

    return model_configs

MODEL_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), 'model_configs')
@cache
def load_all_configs() -> dict[str, FullModelConfig]:
    model_configs: dict[str, FullModelConfig] = {}
    for file in os.listdir(MODEL_CONFIGS_DIR):
        model_configs.update(load_model_configs(os.path.join(MODEL_CONFIGS_DIR, file)))
    return model_configs

@cache
def get_model_config(model_name: str) -> FullModelConfig:
    model_configs = load_all_configs()
    matches: list[FullModelConfig] = []
    for model_config in model_configs.values():
        if model_name.lower() == model_config.display_name.lower() or (model_config.aliases and model_name.lower() in [alias.lower() for alias in model_config.aliases]):
            matches.append(model_config)
    if len(matches) > 1:
        raise ValueError(f"Multiple model configs found for {model_name}")
    elif len(matches) == 0:
        return FullModelConfig(
            display_name=model_name.lower(),
            api_name={
                'local': model_name
            },
            aliases=[],
            api_kwargs={}
        )
    else:
        return matches[0]

def prepare_model_config(
    model_name: str,
    api_dict: dict | None = None, 
    force_max_tokens: int | None = None, 
    force_temperature: float | None = None, 
    model_provider_override: str | None = None,
    model_display_name_override: str | None = None
    ):
    model_config = get_model_config(model_name)

    model_display_name = model_display_name_override if model_display_name_override else model_config.display_name

    provider = model_provider_override
    if provider is None and api_dict is not None:
        assert 'api_base' in api_dict, "Missing API base for model"
        provider = 'local'
    if provider is None:
        provider = model_config.default_provider
    if provider is None:
        provider = list(model_config.api_name.keys())[0]

    if 'https' in provider:
        # provider name is a base URL
        if api_dict is None:
            api_dict = {
                'api_base': provider,   
            }
            if model_config.api_keys and os.environ.get(model_config.api_keys[provider]):
                api_dict['api_key'] = os.environ.get(model_config.api_keys[provider])
    
    if provider is None:
        raise ValueError(f"Missing provider for model {model_config.display_name}")
    
    model_api_name = model_config.api_name[provider]

    api_kwargs = {}
    if model_config.api_kwargs:
        if model_config.api_kwargs.get('default'):
            # pull default kwargs for model
            api_kwargs = model_config.api_kwargs['default']
        if provider in model_config.api_kwargs:
            # update with provider-specific kwargs
            api_kwargs.update(model_config.api_kwargs[provider])

    if provider == 'local':
        assert api_dict is not None, "Missing API dict for local model"
        assert 'api_base' in api_dict, "Missing API base for local model"

    agent_config = AgentConfig()
    if model_config.agent_config is not None:
        if 'default' in model_config.agent_config:
            agent_config = model_config.agent_config['default']
        if provider in model_config.agent_config:
            agent_config.update(model_config.agent_config[provider])

    if force_temperature:
        api_kwargs['temperature'] = force_temperature
    elif 'temperature' not in model_config.api_kwargs:
        api_kwargs['temperature'] = DEFAULT_TEMPERATURE
    if force_max_tokens:
        api_kwargs['max_tokens'] = force_max_tokens
    elif all(x not in model_config.api_kwargs for x in ['max_tokens', 'max_output_tokens', 'max_completion_tokens']):
        api_kwargs['max_tokens'] = DEFAULT_MAX_TOKENS

    return ModelConfig(
        display_name=model_display_name, 
        api_name=model_api_name, 
        api_provider=provider, 
        api_dict=api_dict, 
        api_kwargs=api_kwargs,
        prompt_prefix=model_config.prompt_prefix, 
        agent_config=agent_config
    )