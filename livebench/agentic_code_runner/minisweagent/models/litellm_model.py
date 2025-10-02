import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import litellm
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from livebench.agentic_code_runner.minisweagent.models import GLOBAL_MODEL_STATS
from livebench.agentic_code_runner.minisweagent.utils.log import logger


@dataclass
class LitellmModelConfig:
    model_name: str
    api_type: Literal["completion", "responses"] = "completion"
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    litellm_model_registry: Path | str | None = os.getenv("LITELLM_MODEL_REGISTRY_PATH")


class LitellmModel:
    def __init__(self, **kwargs):
        self.config = LitellmModelConfig(**kwargs)
        self.cost = 0.0
        self.n_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        if self.config.litellm_model_registry and Path(self.config.litellm_model_registry).is_file():
            litellm.utils.register_model(json.loads(Path(self.config.litellm_model_registry).read_text()))

    @retry(
        stop=stop_after_attempt(15),
        wait=wait_exponential(multiplier=2, min=4, max=120),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(
            (
                litellm.exceptions.UnsupportedParamsError,
                litellm.exceptions.NotFoundError,
                litellm.exceptions.PermissionDeniedError,
                litellm.exceptions.ContextWindowExceededError,
                litellm.exceptions.AuthenticationError,
                KeyboardInterrupt,
            )
        ),
    )
    def _query_completion(self, messages: list[dict[str, str]], **kwargs):
        if 'deepseek' in self.config.model_name:
            for message in messages:
                if message['role'] == 'system':
                    message['role'] = 'user'
        res = litellm.completion(
            model=self.config.model_name, messages=messages, **(self.config.model_kwargs | kwargs)
        )

        # return res, res.choices[0].message.content if res and res.choices and len(res.choices) > 0 else None

        result = {
            'response': res
        }
        if res and res.choices and len(res.choices) > 0:
            result['content'] = res.choices[0].message.content
            result['input_tokens'] = res.usage.prompt_tokens
            result['output_tokens'] = res.usage.completion_tokens
        return result

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(
            (
                litellm.exceptions.UnsupportedParamsError,
                litellm.exceptions.NotFoundError,
                litellm.exceptions.PermissionDeniedError,
                litellm.exceptions.ContextWindowExceededError,
                litellm.exceptions.AuthenticationError,
                KeyboardInterrupt,
            )
        ),
    )
    def _query_responses(self, messages: list[dict[str, str]], **kwargs):
        system_messages = [m for m in messages if m['role'] == 'system']
        system_prompt = system_messages[0]['content'] if system_messages else None
        actual_messages = [m.copy() for m in messages if m['role'] != 'system']
        for message in actual_messages:
            if 'extra' in message:
                del message['extra']
        res = litellm.responses(
            model=self.config.model_name, input=actual_messages, instructions=system_prompt, **(self.config.model_kwargs | kwargs),
        )

        output_text = ""

        for output_item in res.output:
            if output_item.type == 'message':
                for content in output_item.content:
                    if content.type == 'output_text':
                        output_text += content.text or ""

        result = {
            'response': res,
            'content': output_text
        }
        if res and res.usage is not None:
            result['input_tokens'] = res.usage.input_tokens
            result['output_tokens'] = res.usage.output_tokens
        return result

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        if self.config.api_type == "completion":
            result = self._query_completion(messages, **kwargs)
        elif self.config.api_type == "responses":
            result = self._query_responses(messages, **kwargs)
        else:
            raise ValueError(f"Invalid API type: {self.config.api_type}")
        response = result['response']
        content = result['content']
        input_tokens = result['input_tokens']
        output_tokens = result['output_tokens']
        try:
            cost = litellm.cost_calculator.completion_cost(response)
            self.cost += cost
            GLOBAL_MODEL_STATS.add(cost)
        except Exception as e:
            logger.warning(
                f"Error calculating cost for model {self.config.model_name}: {e}. "
                "Please check the 'Updating the model registry' section in the documentation at "
                "https://klieret.short.gy/litellm-model-registry Still stuck? Please open a github issue for help!"
            )
        self.n_calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        return {
            "content": content or "",
            "extra": {
                "response": response.model_dump(),
            },
        }

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config) | {"n_model_calls": self.n_calls, "model_cost": self.cost, "total_input_tokens": self.input_tokens, "total_output_tokens": self.output_tokens}
