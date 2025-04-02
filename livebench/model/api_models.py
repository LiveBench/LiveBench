import sys

from livebench.model.completions import chat_completion_openai, chat_completion_together
from livebench.model.model_adapter import get_model_adapter
from livebench.model.models import (AnthropicModel, AWSModel, CohereModel,
                                    DeepseekModel, GeminiModel, GemmaModel,
                                    LlamaModel, MistralModel, Model,
                                    NvidiaModel, OpenAIModel, OpenAIResponsesModel, PerplexityModel,
                                    QwenModel, QwenModelAlibabaAPI, XAIModel, StepFunModel)

if sys.version_info >= (3, 9):
    from functools import cache
else:
    from functools import lru_cache as cache


# Anthropic Models
ANTHROPIC_MODELS = [
    AnthropicModel(api_name="claude-1", display_name="claude-1", aliases=[]),
    AnthropicModel(api_name="claude-2", display_name="claude-2", aliases=[]),
    AnthropicModel(api_name="claude-2.0", display_name="claude-2.0", aliases=[]),
    AnthropicModel(api_name="claude-2.1", display_name="claude-2.1", aliases=[]),
    AnthropicModel(
        api_name="claude-instant-1", display_name="claude-instant-1", aliases=[]
    ),
    AnthropicModel(
        api_name="claude-instant-1.2", display_name="claude-instant-1.2", aliases=[]
    ),
    AnthropicModel(
        api_name="claude-3-opus-20240229",
        display_name="claude-3-opus-20240229",
        aliases=['claude-3-opus'],
    ),
    AnthropicModel(
        api_name="claude-3-sonnet-20240229",
        display_name="claude-3-sonnet-20240229",
        aliases=[],
    ),
    AnthropicModel(
        api_name="claude-3-haiku-20240307",
        display_name="claude-3-haiku-20240307",
        aliases=[],
    ),
    AnthropicModel(
        api_name="claude-3-5-sonnet-20240620",
        display_name="claude-3-5-sonnet-20240620",
        aliases=[],
    ),
    AnthropicModel(
        api_name="claude-3-5-sonnet-20241022",
        display_name="claude-3-5-sonnet-20241022",
        aliases=['claude-3-5-sonnet'],
    ),
    AnthropicModel(
        api_name="claude-3-5-haiku-20241022",
        display_name="claude-3-5-haiku-20241022",
        aliases=['claude-3-5-haiku'],
    ),
    AnthropicModel(
        api_name="claude-3-7-sonnet-20250219",
        display_name="claude-3-7-sonnet-20250219-thinking-25k",
        aliases=['claude-3-7-sonnet-thinking-25k'],
        api_kwargs={
            'thinking': {
                'type': 'enabled',
                'budget_tokens': 25000
            },
            'max_tokens': 32000,
            'temperature': None
        }
    ),
    AnthropicModel(
        api_name="claude-3-7-sonnet-20250219",
        display_name="claude-3-7-sonnet-20250219-thinking-64k",
        aliases=['claude-3-7-sonnet-thinking-64k', 'claude-3-7-sonnet-thinking'],
        api_kwargs={
            'thinking': {
                'type': 'enabled',
                'budget_tokens': 63000
            },
            'max_tokens': 64000,
            'temperature': None
        }
    ),
    AnthropicModel(
        api_name="claude-3-7-sonnet-20250219",
        display_name="claude-3-7-sonnet-20250219-base",
        aliases=['claude-3-7-sonnet-base', 'claude-3-7-sonnet'],
    ),
]

# OpenAI Models
OPENAI_MODELS = [
    OpenAIModel(api_name="gpt-3.5-turbo", display_name="gpt-3.5-turbo", aliases=[]),
    OpenAIModel(
        api_name="gpt-3.5-turbo-0301", display_name="gpt-3.5-turbo-0301", aliases=[]
    ),
    OpenAIModel(
        api_name="gpt-3.5-turbo-0613", display_name="gpt-3.5-turbo-0613", aliases=[]
    ),
    OpenAIModel(
        api_name="gpt-3.5-turbo-1106", display_name="gpt-3.5-turbo-1106", aliases=[]
    ),
    OpenAIModel(
        api_name="gpt-3.5-turbo-0125", display_name="gpt-3.5-turbo-0125", aliases=[]
    ),
    OpenAIModel(api_name="gpt-4", display_name="gpt-4", aliases=[]),
    OpenAIModel(api_name="gpt-4-0314", display_name="gpt-4-0314", aliases=[]),
    OpenAIModel(api_name="gpt-4-0613", display_name="gpt-4-0613", aliases=[]),
    OpenAIModel(api_name="gpt-4-turbo", display_name="gpt-4-turbo", aliases=[]),
    OpenAIModel(
        api_name="gpt-4-turbo-2024-04-09",
        display_name="gpt-4-turbo-2024-04-09",
        aliases=[],
    ),
    OpenAIModel(
        api_name="gpt-4-1106-preview", display_name="gpt-4-1106-preview", aliases=[]
    ),
    OpenAIModel(
        api_name="gpt-4-0125-preview", display_name="gpt-4-0125-preview", aliases=[]
    ),
    OpenAIModel(
        api_name="gpt-4o-2024-05-13", display_name="gpt-4o-2024-05-13", aliases=[]
    ),
    OpenAIModel(
        api_name="gpt-4o-mini-2024-07-18",
        display_name="gpt-4o-mini-2024-07-18",
        aliases=['gpt-4o-mini'],
    ),
    OpenAIModel(
        api_name="gpt-4o-2024-08-06", display_name="gpt-4o-2024-08-06", aliases=[]
    ),
    OpenAIModel(
        api_name='gpt-4o-2024-11-20',
        display_name='gpt-4o-2024-11-20',
        aliases=['gpt-4o']
    ),
    OpenAIModel(
        api_name="chatgpt-4o-latest", display_name="chatgpt-4o-latest-2025-03-27", aliases=['chatgpt-4o-latest']
    ),
    OpenAIModel(
        api_name='gpt-4.5-preview-2025-02-27',
        display_name='gpt-4.5-preview-2025-02-27',
        aliases=['gpt-4.5-preview']
    )
]

INFERENCE_OPENAI_MODELS = [
    OpenAIModel(
        api_name="o1-mini-2024-09-12",
        display_name="o1-mini-2024-09-12",
        aliases=["o1-mini"],
        inference_api=True,
    ),
    OpenAIModel(
        api_name="o1-preview-2024-09-12",
        display_name="o1-preview-2024-09-12",
        aliases=["o1-preview"],
        inference_api=True,
    ),
    OpenAIModel(
        api_name="o1-2024-12-17",
        display_name="o1-2024-12-17-high",
        aliases=['o1', 'o1-high', 'o1-2024-12-17'],
        inference_api=True,
        api_kwargs={'reasoning_effort': 'high', 'use_developer_messages': True}
    ),
    OpenAIModel(
        api_name="o1-2024-12-17",
        display_name="o1-2024-12-17-low",
        aliases=['o1-low'],
        inference_api=True,
        api_kwargs={'reasoning_effort': 'low'}
    ),
    OpenAIModel(
        api_name="o1-2024-12-17",
        display_name="o1-2024-12-17-medium",
        aliases=['o1-medium'],
        inference_api=True,
        api_kwargs={'reasoning_effort': 'medium'}
    ),
    OpenAIModel(
        api_name='o3-mini-2025-01-31',
        display_name='o3-mini-2025-01-31-high',
        aliases=['o3-mini-high', 'o3-mini', 'o3-mini-2025-01-31'],
        inference_api=True,
        api_kwargs={'reasoning_effort': 'high'}
    ),
    OpenAIModel(
        api_name='o3-mini-2025-01-31',
        display_name='o3-mini-2025-01-31-low',
        aliases=['o3-mini-low'],
        inference_api=True,
        api_kwargs={'reasoning_effort': 'low'}
    ),
    OpenAIModel(
        api_name='o3-mini-2025-01-31',
        display_name='o3-mini-2025-01-31-medium',
        aliases=['o3-mini-medium'],
        inference_api=True,
        api_kwargs={'reasoning_effort': 'medium'}
    ),
    OpenAIResponsesModel(
        api_name='o1-pro-2025-03-19',
        display_name='o1-pro-2025-03-19',
        aliases=['o1-pro'],
        inference_api=True,
        api_kwargs={'reasoning_effort': 'high'}
    )
]

# Together Models
TOGETHER_MODELS = [
    LlamaModel(
        api_name="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        display_name="meta-llama-3.1-405b-instruct-turbo",
        aliases=[],
    ),
    LlamaModel(
        api_name="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        display_name="meta-llama-3.1-70b-instruct-turbo",
        aliases=[],
    ),
    LlamaModel(
        api_name="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        display_name="meta-llama-3.1-8b-instruct-turbo",
        aliases=[],
    ),
    LlamaModel(
        api_name="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        display_name="llama-3.3-70b-instruct-turbo",
        aliases=[],
    ),
    QwenModel(
        api_name="Qwen/Qwen2.5-7B-Instruct-Turbo",
        display_name="qwen2.5-7b-instruct-turbo",
        aliases=['qwen2.5-7b-instruct'],
    ),
    QwenModel(
        api_name="Qwen/Qwen2.5-72B-Instruct-Turbo",
        display_name="qwen2.5-72b-instruct-turbo",
        aliases=['qwen2.5-72b-instruct'],
    ),
    QwenModel(
        api_name="Qwen/QwQ-32B-Preview",
        display_name="qwq-32b-preview",
        aliases=[],
        api_kwargs={'max_tokens': 16000, 'temperature': 0.7, 'top_p': 0.95}
    ),
    QwenModel(
        api_name="Qwen/QwQ-32B",
        display_name="qwq-32b",
        aliases=[],
        api_kwargs={'max_tokens': 31000, 'temperature': 0.7, 'top_p': 0.95}
    ),
    LlamaModel(
        api_name="Llama-3.1-Nemotron-70B-Instruct-HF",
        display_name="llama-3.1-nemotron-70b-instruct",
        aliases=['llama-3.1-nemotron-70b-instruct', 'nvidia/llama-3.1-nemotron-70b-instruct'],
    ),
    GemmaModel(
        api_name="google/gemma-2-27b-it", display_name="gemma-2-27b-it", aliases=[]
    ),
    GemmaModel(
        api_name="google/gemma-2-9b-it", display_name="gemma-2-9b-it", aliases=[]
    ),
    DeepseekModel(
        api_name="deepseek-ai/DeepSeek-R1", display_name='deepseek-r1', api_function=chat_completion_together,
        aliases=[], api_kwargs={'temperature': 0.7, 'max_tokens': 20000}
    ),
    DeepseekModel(api_name="deepseek-ai/deepseek-v3", display_name="deepseek-v3-0324", aliases=[], api_function=chat_completion_together)
]

QWEN_ALIBABA_MODELS = [
    QwenModelAlibabaAPI(
        api_name="qwen-max-2025-01-25",
        display_name="qwen2.5-max",
        aliases=[]
    )
]

# Google GenerativeAI Models
GOOGLE_GENERATIVEAI_MODELS = [
    GeminiModel(
        api_name="gemini-1.5-pro-001", display_name="gemini-1.5-pro-001", aliases=[]
    ),
    GeminiModel(
        api_name="gemini-1.5-flash-001", display_name="gemini-1.5-flash-001", aliases=[]
    ),
    GeminiModel(
        api_name="gemini-1.5-pro-exp-0801",
        display_name="gemini-1.5-pro-exp-0801",
        aliases=[],
    ),
    GeminiModel(
        api_name="gemini-1.5-pro-exp-0827",
        display_name="gemini-1.5-pro-exp-0827",
        aliases=[],
    ),
    GeminiModel(
        api_name="gemini-1.5-flash-exp-0827",
        display_name="gemini-1.5-flash-exp-0827",
        aliases=[],
    ),
    GeminiModel(
        api_name="gemini-1.5-flash-8b-exp-0827",
        display_name="gemini-1.5-flash-8b-exp-0827",
        aliases=[],
    ),
    GeminiModel(
        api_name="gemini-1.5-pro-002", display_name="gemini-1.5-pro-002", aliases=[]
    ),
    GeminiModel(
        api_name="gemini-1.5-flash-002", display_name="gemini-1.5-flash-002", aliases=[]
    ),
    GeminiModel(api_name="gemini-exp-1114", display_name="gemini-exp-1114", aliases=[]),
    GeminiModel(api_name="gemini-exp-1121", display_name="gemini-exp-1121", aliases=[]),
    GeminiModel(
        api_name="gemini-1.5-flash-8b-exp-0924",
        display_name="gemini-1.5-flash-8b-exp-0924",
        aliases=[],
    ),
    GeminiModel(
        api_name="learnlm-1.5-pro-experimental",
        display_name="learnlm-1.5-pro-experimental",
        aliases=[],
    ),
    GeminiModel(
        api_name='gemini-exp-1206',
        display_name='gemini-exp-1206',
        aliases=[],
    ),
    GeminiModel(
        api_name='gemini-2.0-flash-exp',
        display_name='gemini-2.0-flash-exp',
        aliases=[],
    ),
    GeminiModel(
        api_name="gemini-2.0-flash-thinking-exp-1219",
        display_name="gemini-2.0-flash-thinking-exp-1219",
        aliases=[],
        api_kwargs={'max_output_tokens': 65536, 'temperature': 0.7, 'top_p': 0.95, 'top_k': 64}
    ),
    GeminiModel(
        api_name="gemini-2.0-flash-thinking-exp-01-21",
        display_name="gemini-2.0-flash-thinking-exp-01-21",
        aliases=['gemini-2.0-flash-thinking-exp'],
        api_kwargs={'max_output_tokens': 65536, 'temperature': 0.7, 'top_p': 0.95, 'top_k': 64, 'thinking_config': {'include_thoughts': True}}
    ),
    GeminiModel(
        api_name='gemini-2.0-pro-exp-02-05',
        display_name='gemini-2.0-pro-exp-02-05',
        aliases=['gemini-2.0-pro-exp'],
    ),
    GeminiModel(
        api_name='gemini-2.0-flash-001',
        display_name='gemini-2.0-flash-001',
        aliases=['gemini-2.0-flash'],
    ),
    GeminiModel(
        api_name='gemini-2.0-flash-lite-preview-02-05',
        display_name='gemini-2.0-flash-lite-preview-02-05',
        aliases=[]
    ),
    GeminiModel(
        api_name='gemini-2.0-flash-lite-001',
        display_name='gemini-2.0-flash-lite-001',
        aliases=['gemini-2.0-flash-lite'],
    ),
    GeminiModel(
        api_name='gemma-3-27b-it',
        display_name='gemma-3-27b-it',
        aliases=['gemma-3-27b']
    ),
    GeminiModel(
        api_name='gemini-2.5-pro-exp-03-25',
        display_name='gemini-2.5-pro-exp-03-25',
        aliases=['gemini-2.5-pro-exp'],
        api_kwargs={'max_output_tokens': 65536, 'temperature': 1, 'top_p': 0.95}
    )
]

# Vertex Models
VERTEX_MODELS = [
    GeminiModel(
        api_name="gemini-1.5-pro-preview-0409",
        display_name="gemini-1.5-pro-preview-0409",
        aliases=[],
    ),
]

# Mistral Models
MISTRAL_MODELS = [
    MistralModel(
        api_name="mistral-large-latest", display_name="mistral-large-latest", aliases=[]
    ),
    MistralModel(
        api_name="mistral-large-2402", display_name="mistral-large-2402", aliases=[]
    ),
    MistralModel(
        api_name="mistral-medium-23-12", display_name="mistral-medium-23-12", aliases=[]
    ),
    MistralModel(api_name="mistral-medium", display_name="mistral-medium", aliases=[]),
    MistralModel(
        api_name="mistral-small-2402", display_name="mistral-small-2402", aliases=[]
    ),
    MistralModel(
        api_name="open-mixtral-8x7b", display_name="open-mixtral-8x7b", aliases=[]
    ),
    MistralModel(
        api_name="open-mixtral-8x22b", display_name="open-mixtral-8x22b", aliases=[]
    ),
    MistralModel(
        api_name="mistral-large-2407", display_name="mistral-large-2407", aliases=[]
    ),
    MistralModel(
        api_name="open-mistral-nemo", display_name="open-mistral-nemo", aliases=[]
    ),
    MistralModel(
        api_name="mistral-large-2411", display_name="mistral-large-2411", aliases=[]
    ),
    MistralModel(
        api_name="mistral-small-2409", display_name="mistral-small-2409", aliases=[]
    ),
    MistralModel(
        api_name="mistral-small-2501", display_name="mistral-small-2501", aliases=[]
    ),
    MistralModel(
        api_name="mistral-small-2503", display_name="mistral-small-2503", aliases=['mistral-small']
    )
]

# Cohere Models
COHERE_MODELS = [
    CohereModel(
        api_name="command-r-plus-04-2024",
        display_name="command-r-plus-04-2024",
        aliases=[],
    ),
    CohereModel(
        api_name="command-r-03-2024", display_name="command-r-03-2024", aliases=[]
    ),
    CohereModel(api_name="command", display_name="command", aliases=[]),
    CohereModel(
        api_name="command-r-08-2024", display_name="command-r-08-2024", aliases=[]
    ),
    CohereModel(
        api_name="command-r-plus-08-2024",
        display_name="command-r-plus-08-2024",
        aliases=[],
    ),
]

# Deepseek Models
DEEPSEEK_MODELS = [
    #DeepseekModel(api_name="deepseek-chat", display_name="deepseek-v3-0324", aliases=[]),
    # DeepseekModel(api_name="deepseek-reasoner", display_name="deepseek-r1", aliases=[], reasoner=True)
]

# Nvidia Models
NVIDIA_MODELS = [
    NvidiaModel(
        api_name="nvidia/nemotron-4-340b-instruct",
        display_name="nemotron-4-340b-instruct",
        aliases=[],
    ),
]

# XAI Models
XAI_MODELS = [
    XAIModel(api_name="grok-beta", display_name="grok-beta", aliases=[]),
    XAIModel(api_name="grok-2-1212", display_name="grok-2-1212", aliases=[]),
]

# AWS Models
AWS_MODELS = [
    AWSModel(
        api_name="amazon.nova-micro-v1:0",
        display_name="amazon.nova-micro-v1:0",
        aliases=[],
    ),
    AWSModel(
        api_name="amazon.nova-micro-v1:0:128k",
        display_name="amazon.nova-micro-v1:0:128k",
        aliases=[],
    ),
    AWSModel(
        api_name="amazon.nova-lite-v1:0",
        display_name="amazon.nova-lite-v1:0",
        aliases=[],
    ),
    AWSModel(
        api_name="amazon.nova-lite-v1:0:300k",
        display_name="amazon.nova-lite-v1:0:300k",
        aliases=[],
    ),
    AWSModel(
        api_name="amazon.nova-pro-v1:0", display_name="amazon.nova-pro-v1:0", aliases=[]
    ),
    AWSModel(
        api_name="amazon.nova-pro-v1:0:300k",
        display_name="amazon.nova-pro-v1:0:300k",
        aliases=[],
    ),
]


# Perplexity Models
PERPLEXITY_MODELS = [
    PerplexityModel(api_name="sonar", display_name="sonar", aliases=[]),
    PerplexityModel(api_name="sonar-pro", display_name="sonar-pro", aliases=[]),
    PerplexityModel(api_name="sonar-reasoning", display_name="sonar-reasoning", aliases=[]),
    PerplexityModel(api_name="sonar-reasoning-pro", display_name="sonar-reasoning-pro", aliases=[]),
]


STEPFUN_MODELS = [
    StepFunModel(
        api_name='step-2-16k-202411',
        display_name='step-2-16k-202411',
        aliases=['step-2-16k']
    )
]

ALL_MODELS = (
    ANTHROPIC_MODELS
    + OPENAI_MODELS
    + INFERENCE_OPENAI_MODELS
    + MISTRAL_MODELS
    + COHERE_MODELS
    + DEEPSEEK_MODELS
    + NVIDIA_MODELS
    + XAI_MODELS
    + AWS_MODELS
    + PERPLEXITY_MODELS
    + GOOGLE_GENERATIVEAI_MODELS
    + QWEN_ALIBABA_MODELS
    + TOGETHER_MODELS
    + STEPFUN_MODELS
)


@cache
def get_model(name: str) -> Model:
    matches = []
    for model in ALL_MODELS:
        if (
            model.display_name.lower() == name.lower()
            or any(alias.lower() == name.lower() for alias in model.aliases)
        ):
            matches.append(model)
    if len(matches) > 1:
        raise ValueError(f"Multiple models found for {name}")
    elif len(matches) == 0:
        # warnings.warn(f"No model found for {name}")
        return Model(
            api_name=name,
            display_name=name,
            aliases=[],
            adapter=get_model_adapter(name),
            api_function=chat_completion_openai,
        )
    else:
        return matches[0]
