"""Environment implementations for mini-SWE-agent."""

import copy
import importlib

from livebench.agentic_code_runner.minisweagent import Environment

_ENVIRONMENT_MAPPING = {
    "docker": "livebench.agentic_code_runner.minisweagent.environments.docker.DockerEnvironment",
    "local": "livebench.agentic_code_runner.minisweagent.environments.local.LocalEnvironment",
}


def get_environment_class(spec: str) -> type[Environment]:
    full_path = _ENVIRONMENT_MAPPING.get(spec, spec)
    try:
        module_name, class_name = full_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ValueError, ImportError, AttributeError):
        msg = f"Unknown environment type: {spec} (resolved to {full_path}, available: {_ENVIRONMENT_MAPPING})"
        raise ValueError(msg)


def get_environment(config: dict, *, cwd_override: str = "", default_type: str = "") -> Environment:
    config = copy.deepcopy(config)
    if cwd_override:
        config["cwd"] = cwd_override
    environment_class = config.pop("environment_class", default_type)
    return get_environment_class(environment_class)(**config)
