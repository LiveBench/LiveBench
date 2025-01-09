import os
from collections.abc import Callable
from dataclasses import dataclass, field

from livebench.conversation import Conversation
from livebench.model.completions import (chat_completion_anthropic,
                                         chat_completion_aws,
                                         chat_completion_cohere,
                                         chat_completion_deepseek,
                                         chat_completion_google_generativeai,
                                         chat_completion_mistral,
                                         chat_completion_nvidia,
                                         chat_completion_openai, chat_completion_openai_responses,
                                         chat_completion_perplexity,
                                         chat_completion_together,
                                         chat_completion_xai)
from livebench.model.model_adapter import (BaseModelAdapter, ChatGPTAdapter,
                                           ClaudeAdapter, CohereAdapter,
                                           DeepseekChatAdapter, GeminiAdapter,
                                           GemmaAdapter, Llama3Adapter,
                                           MistralAdapter, NvidiaChatAdapter,
                                           QwenChatAdapter)

model_api_function = Callable[["Model", Conversation, float, int, dict | None], tuple[str, int]]


@dataclass(kw_only=True, frozen=True)
class Model:
    api_name: str
    display_name: str
    aliases: list[str]
    adapter: BaseModelAdapter
    api_function: model_api_function | None = None
    api_kwargs: dict = field(default_factory=dict)


@dataclass(kw_only=True, frozen=True)
class AnthropicModel(Model):
    adapter: BaseModelAdapter = field(default=ClaudeAdapter())
    api_function: model_api_function = field(
        default=chat_completion_anthropic
    )


@dataclass(kw_only=True, frozen=True)
class OpenAIModel(Model):
    adapter: BaseModelAdapter = field(default=ChatGPTAdapter())
    inference_api: bool = False
    api_function: model_api_function = field(
        default=chat_completion_openai
    )

@dataclass(kw_only=True, frozen=True)
class OpenAIResponsesModel(OpenAIModel):
    api_function: model_api_function = field(
        default=chat_completion_openai_responses
    )


@dataclass(kw_only=True, frozen=True)
class LlamaModel(Model):
    adapter: BaseModelAdapter = field(default=Llama3Adapter())
    api_function: model_api_function = field(
        default=chat_completion_together
    )

@dataclass(kw_only=True, frozen=True)
class QwenModelAlibabaAPI(Model):
    adapter: BaseModelAdapter = field(default=QwenChatAdapter())
    api_function: model_api_function = field(
        default=lambda model, conv, temperature, max_tokens, api_dict: chat_completion_openai(model, conv, temperature, max_tokens, {'api_base': 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1', 'api_key': os.environ.get('QWEN_API_KEY', None)})
    )

@dataclass(kw_only=True, frozen=True)
class QwenModel(Model):
    adapter: BaseModelAdapter = field(default=QwenChatAdapter())
    api_function: model_api_function = field(
        default=chat_completion_together
    )


@dataclass(kw_only=True, frozen=True)
class GemmaModel(Model):
    adapter: BaseModelAdapter = field(default=GemmaAdapter())
    api_function: model_api_function = field(
        default=chat_completion_together
    )


@dataclass(kw_only=True, frozen=True)
class GeminiModel(Model):
    adapter: BaseModelAdapter = field(default=GeminiAdapter())
    api_function: model_api_function = field(
        default=chat_completion_google_generativeai
    )


@dataclass(kw_only=True, frozen=True)
class MistralModel(Model):
    adapter: BaseModelAdapter = field(default=MistralAdapter())
    api_function: model_api_function = field(
        default=chat_completion_mistral
    )


@dataclass(kw_only=True, frozen=True)
class CohereModel(Model):
    adapter: BaseModelAdapter = field(default=CohereAdapter())
    api_function: model_api_function = field(
        default=chat_completion_cohere
    )


@dataclass(kw_only=True, frozen=True)
class DeepseekModel(Model):
    adapter: BaseModelAdapter = field(default=DeepseekChatAdapter())
    api_function: model_api_function = field(
        default=chat_completion_deepseek
    )
    reasoner: bool = field(default=False)


@dataclass(kw_only=True, frozen=True)
class NvidiaModel(Model):
    adapter: BaseModelAdapter = field(default=NvidiaChatAdapter())
    api_function: model_api_function = field(
        default=chat_completion_nvidia
    )


@dataclass(kw_only=True, frozen=True)
class XAIModel(Model):
    adapter: BaseModelAdapter = field(default=ChatGPTAdapter())
    api_function: model_api_function = field(
        default=chat_completion_xai
    )


@dataclass(kw_only=True, frozen=True)
class AWSModel(Model):
    adapter: BaseModelAdapter = field(default=ChatGPTAdapter())
    api_function: model_api_function = field(
        default=chat_completion_aws
    )


@dataclass(kw_only=True, frozen=True)
class PerplexityModel(Model):
    adapter: BaseModelAdapter = field(default=ChatGPTAdapter())
    api_function: model_api_function = field(
        default=chat_completion_perplexity
    )
