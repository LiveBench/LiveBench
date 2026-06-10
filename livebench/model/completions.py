import logging
import os
import sys
import time
import traceback
from typing import Any, Protocol, cast

import httpx
from openai import Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_fixed, wait_incrementing)

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)


logger = logging.getLogger(__name__)

# API setting constants
API_MAX_RETRY = 1
API_RETRY_SLEEP_MIN = 10
API_RETRY_SLEEP_MAX = 60
API_ERROR_OUTPUT = "$ERROR$"
TIMEOUT = 1800

# model api function takes in Model, list of messages, temperature, max tokens, api kwargs, and an api dict
# returns tuple of (output, num tokens)
Conversation = list[dict[str, str]]
API_Kwargs = dict[str, str | float | int | dict[str, str] | None]

class ModelAPI(Protocol):
    def __call__(self, model: str, messages: Conversation, temperature: float, max_tokens: int, model_api_kwargs: API_Kwargs | None, api_dict: dict[str, str] | None, stream: bool) -> tuple[str, int, dict[str, Any] | None]:
        ...


def retry_fail(retry_state):
    exception = retry_state.outcome.exception()
    error = exception.__class__.__name__ if exception else "Unknown"
    error_msg = str(exception)[:200] if exception else ""
    print(f"all retries failed: {error}: {error_msg}")
    return API_ERROR_OUTPUT, 0, {"eval_status": "api_error", "error": error, "error_msg": error_msg}


def retry_log(retry_state):
    exception = retry_state.outcome.exception()
    logger.warning(f"{retry_state.fn.__name__}: attempt {retry_state.attempt_number} failed with {exception.__class__.__name__}: {exception}; {retry_state.seconds_since_start} seconds elapsed total")
    logger.warning("Exception stack trace:", exc_info=exception)


@retry(
    stop=stop_after_attempt(API_MAX_RETRY),
    wait=wait_fixed(API_RETRY_SLEEP_MIN),
    retry=retry_if_exception_type(Exception),
    after=retry_log,
    retry_error_callback=retry_fail
)
def chat_completion_openai(
    model: str, messages: Conversation, temperature: float, max_tokens: int, model_api_kwargs: API_Kwargs | None = None, api_dict: dict[str, str] | None = None, stream: bool = False
) -> tuple[str, int, dict[str, Any] | None]:
    from openai import NOT_GIVEN, OpenAI

    if api_dict is not None:
        client = OpenAI(
            api_key=api_dict["api_key"], base_url=api_dict["api_base"], timeout=httpx.Timeout(timeout=TIMEOUT, connect=10.0)
        )

    else:
        client = OpenAI(timeout=1000)

    api_kwargs: API_Kwargs = {
        'temperature': temperature
    }

    if model_api_kwargs is not None:
        model_api_kwargs = {key: value for key, value in model_api_kwargs.items()}
        api_kwargs.update(model_api_kwargs)

    if 'max_tokens' not in api_kwargs and 'max_completion_tokens' not in api_kwargs:
        # gpt models can use max_completion_tokens but other apis might not support this
        api_kwargs['max_completion_tokens'] = max_tokens if 'gpt' in model else None
        api_kwargs['max_tokens'] = max_tokens if 'gpt' not in model else None

    actual_api_kwargs = {key: (value if value is not None else NOT_GIVEN) for key, value in api_kwargs.items()}

    if 'stream' in actual_api_kwargs:
        stream = actual_api_kwargs['stream']
        del actual_api_kwargs['stream']

    try:
        if stream:
            stream: Stream[ChatCompletionChunk] = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                stream_options={'include_usage': True},
                **actual_api_kwargs
            )

            message = ''
            num_tokens = None
            input_tokens = None
            cached_tokens = None
            try:
                for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
                        message += chunk.choices[0].delta.content
                    if chunk.usage is not None:
                        num_tokens = chunk.usage.completion_tokens
                        if hasattr(chunk.usage, 'reasoning_tokens'):
                            num_tokens += chunk.usage.reasoning_tokens
                        input_tokens = chunk.usage.prompt_tokens
                        if hasattr(chunk.usage, 'prompt_tokens_details') and chunk.usage.prompt_tokens_details is not None:
                            cached_tokens = chunk.usage.prompt_tokens_details.cached_tokens
            except Exception:
                if message != '':
                    print(message)
                raise
        else:
            response: ChatCompletion = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
                **actual_api_kwargs
            )
            if response is None:
                raise Exception("No response returned from OpenAI")
            elif response.choices is None:
                print(response)
                raise Exception("API request failed")
            if isinstance(response.choices[0], str):
                message = response.choices[0]
            else:
                message = response.choices[0].message.content
            input_tokens = None
            cached_tokens = None
            if response.usage is not None:
                num_tokens = response.usage.completion_tokens
                if hasattr(response.usage, 'completion_tokens_details') and hasattr(response.usage.completion_tokens_details, 'reasoning_tokens'):
                    reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens
                    if num_tokens is not None and reasoning_tokens is not None:
                        num_tokens += reasoning_tokens
                input_tokens = response.usage.prompt_tokens
                if hasattr(response.usage, 'prompt_tokens_details') and response.usage.prompt_tokens_details is not None:
                    cached_tokens = response.usage.prompt_tokens_details.cached_tokens
            else:
                num_tokens = None

            if hasattr(response, 'provider') and response.provider is not None:
                metadata = {
                    'provider': response.provider
                }

            if message is None or message == '':
                print(response)

        if message is None or message == '':
            raise Exception("No message returned from OpenAI")
        if num_tokens is None:
            num_tokens = -1
        output = message

        metadata: dict[str, Any] | None = None
        if input_tokens is not None:
            metadata = {'input_tokens': input_tokens}
            if cached_tokens is not None:
                metadata['cached_tokens'] = cached_tokens

        return output, num_tokens, metadata
    except Exception as e:
        if "invalid_prompt" in str(e).lower():
            print("invalid prompt (model refusal), giving up")
            return API_ERROR_OUTPUT, 0, {"eval_status": "api_error", "error": "InvalidPrompt", "error_msg": str(e)[:200]}
        raise e


def chat_completion_openai_responses(model: str, messages: Conversation, temperature: float, max_tokens: int, model_api_kwargs: API_Kwargs | None = None, api_dict: dict[str, str] | None = None, stream: bool = False) -> tuple[str, int]:
    from openai import NOT_GIVEN, OpenAI

    if api_dict is not None:
        client = OpenAI(
            api_key=api_dict["api_key"], base_url=api_dict["api_base"], timeout=httpx.Timeout(timeout=TIMEOUT, connect=20.0)
        )
    else:
        client = OpenAI(timeout=TIMEOUT)

    messages = [message for message in messages if message['role'] == 'user']
    developer_message = ''
    if 'o1' in model or 'o3' in model or 'o4-mini' in model and 'Formatting reenabled' not in messages[0]['content']:
        developer_message = 'Formatting reenabled\n'

    api_kwargs: API_Kwargs = {
        'max_output_tokens': max_tokens if 'gpt' in model else None,
        'temperature': temperature
    }

    if model_api_kwargs is not None:
        model_api_kwargs = {key: value for key, value in model_api_kwargs.items()}
        api_kwargs.update(model_api_kwargs)

    if 'reasoning_effort' in api_kwargs:
        reasoning = {'effort': api_kwargs['reasoning_effort']}
        api_kwargs['reasoning'] = reasoning
        del api_kwargs['reasoning_effort']

    if 'max_completion_tokens' in api_kwargs:
        api_kwargs['max_output_tokens'] = api_kwargs['max_completion_tokens']
        del api_kwargs['max_completion_tokens']

    actual_api_kwargs = {key: (value if value is not None else NOT_GIVEN) for key, value in api_kwargs.items()}

    input = '\n'.join([message['content'] for message in messages])

    output_text = 'BLOCKED'
    output_tokens = 1
    try:
        response = client.responses.create(
            model=model,
            instructions=developer_message,
            input=input,
            store=False,
            **actual_api_kwargs
        )

        if response is None:
            raise Exception("No response received from OpenAI Responses")
        elif response.output_text is None:
            raise Exception("No output text received from OpenAI Responses")
        elif response.usage is None:
            raise Exception("No usage received from OpenAI Responses")

        output_text = response.output_text
        output_tokens = response.usage.output_tokens

    except Exception as e:
        if 'risk' in str(e):
            pass
        else:
            raise

    return output_text, output_tokens

@retry(
    stop=stop_after_attempt(API_MAX_RETRY),
    wait=wait_fixed(API_RETRY_SLEEP_MIN),
    retry=retry_if_exception_type(Exception),
    after=retry_log,
    retry_error_callback=retry_fail
)
def chat_completion_aws(model: str, messages: Conversation, temperature: float, max_tokens: int, model_api_kwargs: API_Kwargs | None = None, api_dict: dict[str, str] | None = None, stream: bool = False) -> tuple[str, int]:
    import boto3

    # AWS region can be customized via api_dict
    region_name = "us-east-1"
    if api_dict is not None and "region_name" in api_dict:
        region_name = api_dict["region_name"]

    brt = boto3.client("bedrock-runtime", region_name=region_name)
    user_messages = [m['content'] for m in messages if m['role'] == "user"]
    prompt = user_messages[0] if user_messages else ""

    # Set up API kwargs
    inference_config = {
        "maxTokens": max_tokens,
        "temperature": temperature,
    }

    # Update with additional kwargs if provided
    if model_api_kwargs is not None:
        if 'max_tokens' in model_api_kwargs:
            inference_config['maxTokens'] = model_api_kwargs['max_tokens']
        if 'temperature' in model_api_kwargs:
            inference_config['temperature'] = model_api_kwargs['temperature']


    # Make the API call
    response = brt.converse(
        modelId=model,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig=inference_config,
    )

    if response is None:
        raise Exception("No response returned from AWS Bedrock")

    output = response["output"]["message"]["content"][0]["text"]
    num_tokens = response["usage"]["outputTokens"]

    return output, num_tokens


incremental_wait = wait_incrementing(start=API_RETRY_SLEEP_MIN, max=API_RETRY_SLEEP_MAX, increment=20)

def gemini_custom_wait(retry_state):
    if retry_state.outcome.failed:
        exception = retry_state.outcome.exception()
        if exception and "RECITATION" in str(exception) or "MAX_TOKENS" in str(exception):
            return 0.0  # don't wait for recitation or max token errors

    val = incremental_wait(retry_state)
    print(f"Waiting for {val} seconds before retrying attempt {retry_state.attempt_number + 1}")

    # other errors might indicate rate limiting, wait for these
    return val

@retry(
    stop=stop_after_attempt(API_MAX_RETRY),
    wait=gemini_custom_wait,
    retry=retry_if_exception_type(Exception),
    after=retry_log,
    retry_error_callback=retry_fail
)
def chat_completion_google_generativeai(
    model: str, messages: Conversation, temperature: float, max_tokens: int, model_api_kwargs: API_Kwargs | None = None, api_dict: dict[str, str] | None = None, stream: bool = False
) -> tuple[str, int, dict[str, Any] | None]:
    from google import genai
    from google.genai import types

    if api_dict is not None and "api_key" in api_dict:
        api_key = api_dict["api_key"]
    else:
        api_key = os.environ["GEMINI_API_KEY"]

    client = genai.Client(api_key=api_key)

    if any(message['role'] == 'system' for message in messages):
        system = [types.Part.from_text(text=message['content']) for message in messages if message['role'] == "system"][0]
    else:
        system = None
    messages: list[types.Content] = [types.Content(role=message['role'], parts=[types.Part.from_text(text=message['content'])]) for message in messages if message['role'] != 'system']

    safety_settings = [
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY, threshold=types.HarmBlockThreshold.BLOCK_NONE),
    ]

    # Initialize with default kwargs and safety settings
    api_kwargs: API_Kwargs = {
        'safety_settings': safety_settings,
        'system_instruction': system,
        'temperature': temperature,
        'max_output_tokens': max_tokens
    }

    # Update with additional kwargs if provided
    if model_api_kwargs is not None:
        model_api_kwargs = {key: value for key, value in model_api_kwargs.items()}
        api_kwargs.update(model_api_kwargs)

    config = types.GenerateContentConfig(**api_kwargs)

    response = client.models.generate_content(
        model=model,
        contents=messages,
        config=config
    )

    if response is None or response.text is None:
        raise Exception("No response returned from Google")

    message = response.text

    num_tokens = None
    input_tokens = None
    cached_tokens = None
    if response.usage_metadata is not None:
        num_tokens = response.usage_metadata.candidates_token_count
        if response.usage_metadata.thoughts_token_count is not None:
            num_tokens += response.usage_metadata.thoughts_token_count
        input_tokens = response.usage_metadata.prompt_token_count
        cached_tokens = response.usage_metadata.cached_content_token_count

    if num_tokens is None:
        num_tokens = -1

    metadata: dict[str, Any] | None = None
    if input_tokens is not None:
        metadata = {'input_tokens': input_tokens}
        if cached_tokens is not None:
            metadata['cached_tokens'] = cached_tokens

    return message, num_tokens, metadata


@retry(
    stop=stop_after_attempt(API_MAX_RETRY),
    wait=wait_fixed(API_RETRY_SLEEP_MIN),
    retry=retry_if_exception_type(Exception),
    after=retry_log,
    retry_error_callback=retry_fail
)
def chat_completion_together(model: str, messages: Conversation, temperature: float, max_tokens: int, model_api_kwargs: API_Kwargs | None = None, api_dict: dict[str, str] | None = None, stream: bool = False) -> tuple[str, int]:
    from together import Together

    if api_dict is not None and "api_key" in api_dict:
        api_key = api_dict["api_key"]
    else:
        api_key = os.environ["TOGETHER_API_KEY"]
    client = Together(api_key=api_key)


    messages = [message for message in messages if message['role'] == 'user']

    api_kwargs: API_Kwargs = {'max_tokens': max_tokens, 'temperature': temperature}
    if model_api_kwargs is not None:
        model_api_kwargs = {key: value for key, value in model_api_kwargs.items()}
        api_kwargs.update(model_api_kwargs)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        **api_kwargs
    )

    if response is None:
        raise Exception("No response returned from Together AI")
    elif response.choices is None:
        raise Exception("No choices returned from Together AI")

    message = response.choices[0].message.content
    if message is None:
        raise Exception("No message returned from Together AI")

    num_tokens = response.usage.completion_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'completion_tokens') else None

    return message, num_tokens


@retry(
    stop=stop_after_attempt(API_MAX_RETRY),
    wait=wait_fixed(API_RETRY_SLEEP_MIN),
    retry=retry_if_exception_type(Exception),
    after=retry_log,
    retry_error_callback=retry_fail
)
def chat_completion_anthropic(model: str, messages: Conversation, temperature: float, max_tokens: int, model_api_kwargs: API_Kwargs | None = None, api_dict: dict[str, str] | None = None, stream: bool = False) -> tuple[str, int]:
    if api_dict is not None and "api_key" in api_dict:
        api_key = api_dict["api_key"]
    else:
        api_key = os.environ["ANTHROPIC_API_KEY"]

    from anthropic import NOT_GIVEN, Anthropic
    c = Anthropic(api_key=api_key)

    api_kwargs: API_Kwargs = {
        'max_tokens': max_tokens,
        'temperature': temperature
    }


    if model_api_kwargs is not None:
        model_api_kwargs = {key: value for key, value in model_api_kwargs.items()}
        api_kwargs.update(model_api_kwargs)

    actual_api_kwargs = {key: (value if value is not None else NOT_GIVEN) for key, value in api_kwargs.items()}

    system = [message for message in messages if message['role'] == 'system']
    if len(system) > 0:
        system_message = system[0]['content']
    else:
        system_message = None

    messages = [message for message in messages if message['role'] != 'system']

    message = []

    client = c
    if actual_api_kwargs.get('betas', []):
        client = c.beta

    with client.messages.stream(
        model=model,
        messages=messages,
        system=system_message if system_message else NOT_GIVEN,
        **actual_api_kwargs
    ) as stream:
        new_block = {}
        for event in stream:
            if event.type == 'content_block_start':
                new_block = {}
                if event.content_block.type == 'redacted_thinking':
                    new_block['type'] = 'redacted_thinking'
                    new_block['data'] = event.content_block.data
                elif event.content_block.type == 'thinking':
                    new_block['type'] = 'thinking'
                elif event.content_block.type == 'text':
                    new_block['type'] = 'text'
            elif event.type == 'content_block_delta':
                assert new_block['type'] is not None
                if event.delta.type == 'text_delta':
                    if 'text' not in new_block:
                        new_block['text'] = ''
                    new_block['text'] += event.delta.text
                elif event.delta.type == 'thinking_delta':
                    if 'thinking' not in new_block:
                        new_block['thinking'] = ''
                    new_block['thinking'] += event.delta.thinking
                elif event.delta.type == 'signature_delta':
                    new_block['signature'] = event.delta.signature
            elif event.type == 'content_block_stop':
                assert new_block != {}
                for content in new_block:
                    new_block[content] = new_block[content].strip()
                message.append(new_block)
                new_block = {}

    del actual_api_kwargs['max_tokens']
    del actual_api_kwargs['temperature']

    # The stream's final message carries exact usage (input from message_start,
    # output from the final message_delta) — no extra API call needed.
    final_usage = None
    try:
        final_message = stream.get_final_message()
        stop_reason = final_message.stop_reason or 'unknown'
        final_usage = final_message.usage
    except Exception:
        stop_reason = 'unknown'

    text_messages = [c for c in message if c['type'] == 'text' and c.get('text', '').strip()]
    if len(text_messages) == 0:
        block_types = [c['type'] for c in message]
        raise Exception(f"No response from Anthropic (stop_reason={stop_reason}, blocks={block_types})")

    message_text = text_messages[0]['text']

    if final_usage is not None:
        tokens = final_usage.output_tokens
        metadata: dict[str, Any] = {'input_tokens': final_usage.input_tokens}
        cached_tokens = getattr(final_usage, 'cache_read_input_tokens', None)
        if cached_tokens is not None:
            metadata['cached_tokens'] = cached_tokens
        return message_text, tokens, metadata

    # Fallback when usage is unavailable: approximate output tokens via count_tokens
    message_for_counting = [c for c in message if (c['type'] != 'text' or c.get('text', '').strip()) and (c['type'] != 'thinking' or c.get('thinking', '').strip())]
    try:
        tokens = client.messages.count_tokens(
            model=model,
            messages=[
                {"role": "assistant", "content": message_for_counting}
            ],
            **actual_api_kwargs
        ).input_tokens
    except Exception as e:
        print('Failed to count tokens:', e)
        traceback.print_exc()
        tokens = -1

    return message_text, tokens, None


@retry(
    stop=stop_after_attempt(API_MAX_RETRY),
    wait=wait_fixed(API_RETRY_SLEEP_MIN),
    retry=retry_if_exception_type(Exception),
    after=retry_log,
    retry_error_callback=retry_fail
)
def chat_completion_mistral(model: str, messages: Conversation, temperature: float, max_tokens: int, model_api_kwargs: API_Kwargs | None = None, api_dict: dict[str, str] | None = None, stream: bool = False) -> tuple[str, int]:
    if api_dict is not None and "api_key" in api_dict:
        api_key = api_dict["api_key"]
    else:
        api_key = os.environ["MISTRAL_API_KEY"]

    from mistralai import UNSET, Mistral
    client = Mistral(api_key=api_key)

    # Set up API kwargs
    api_kwargs: API_Kwargs = {
        'max_tokens': max_tokens,
        'temperature': temperature
    }
    if model_api_kwargs is not None:
        model_api_kwargs = {key: value for key, value in model_api_kwargs.items()}
        api_kwargs.update(model_api_kwargs)

    actual_api_kwargs = {key: (value if value is not None else UNSET) for key, value in api_kwargs.items()}

    chat_response = client.chat.complete(
        model=model,
        messages=messages,
        **actual_api_kwargs
    )

    if chat_response is None:
        raise Exception("No response returned from Mistral")
    elif not hasattr(chat_response, 'choices') or not chat_response.choices:
        raise Exception("No choices returned from Mistral")

    message = chat_response.choices[0].message.content
    if message is None:
        raise Exception("No message returned from Mistral")

    num_tokens = chat_response.usage.completion_tokens if hasattr(chat_response, 'usage') and hasattr(chat_response.usage, 'completion_tokens') else None

    return message.strip(), num_tokens

@retry(
    stop=stop_after_attempt(API_MAX_RETRY),
    wait=wait_fixed(API_RETRY_SLEEP_MIN),
    retry=retry_if_exception_type(Exception),
    after=retry_log,
    retry_error_callback=retry_fail
)
def individual_completion_deepinfra(client, model, messages, ai_message, actual_api_kwargs):
    actual_messages = messages + [ai_message] if ai_message else messages
    response = client.chat.completions.create(
        model=model,
        messages=actual_messages,
        stream=False,
        **actual_api_kwargs
    )

    if response is None:
        raise Exception("No response returned from DeepInfra")
    elif response.choices is None or len(response.choices) == 0:
        print(response)
        raise Exception("No choices returned from DeepInfra")
    elif response.choices[0].message is None or response.choices[0].message.content is None:
        print(response)
        raise Exception("No message returned from DeepInfra")

    return response


def chat_completion_deepinfra(model: str, messages: Conversation, temperature: float, max_tokens: int, model_api_kwargs: API_Kwargs | None = None, api_dict: dict[str, str] | None = None, stream: bool = False) -> tuple[str, int]:
    from openai import NOT_GIVEN, OpenAI

    if api_dict is not None and "api_key" in api_dict:
        api_key = api_dict["api_key"]
    else:
        api_key = os.environ["DEEPINFRA_API_KEY"]

    client = OpenAI(api_key=api_key, base_url="https://api.deepinfra.com/v1/openai", timeout=httpx.Timeout(timeout=TIMEOUT, connect=10.0))

    api_kwargs: API_Kwargs = {
        'max_tokens': max_tokens,
        'temperature': temperature
    }

    if model_api_kwargs is not None:
        model_api_kwargs = {key: value for key, value in model_api_kwargs.items()}
        api_kwargs.update(model_api_kwargs)

    actual_api_kwargs = {key: (value if value is not None else NOT_GIVEN) for key, value in api_kwargs.items()}

    ai_message = None
    total_tokens = 0
    # DeepInfra hard caps at 16384 tokens in a single request
    # so we need to resubmit the request until we get a 'stop' finish reason
    # or reach our actual max tokens
    while total_tokens < actual_api_kwargs['max_tokens']:
        actual_api_kwargs['max_tokens'] = actual_api_kwargs['max_tokens'] - total_tokens

        response = individual_completion_deepinfra(client, model, messages, ai_message, actual_api_kwargs)
        if ai_message is None:
            ai_message = {'role': 'assistant', 'content': ''}
        ai_message['content'] += response.choices[0].message.content
        total_tokens += cast(int, response.usage.completion_tokens)

        if response.choices[0].finish_reason != 'length':
            break
        elif total_tokens < actual_api_kwargs['max_tokens']:
            print(f"Continuing DeepInfra request for more tokens, have {total_tokens} and requested {actual_api_kwargs['max_tokens']}")

    return ai_message['content'], total_tokens

@retry(
    stop=stop_after_attempt(API_MAX_RETRY),
    wait=wait_fixed(API_RETRY_SLEEP_MIN),
    retry=retry_if_exception_type(Exception),
    after=retry_log,
    retry_error_callback=retry_fail
)
def chat_completion_litellm(
    model: str, messages: Conversation, temperature: float, max_tokens: int, model_api_kwargs: API_Kwargs | None = None, api_dict: dict[str, str] | None = None, stream: bool = False, provider: str | None = None
) -> tuple[str, int, dict[str, Any] | None]:
    """Provider-agnostic path via LiteLLM, selected by the --use-litellm flag."""
    import litellm

    api_kwargs: API_Kwargs = {
        'temperature': temperature
    }

    if model_api_kwargs is not None:
        model_api_kwargs = {key: value for key, value in model_api_kwargs.items()}
        api_kwargs.update(model_api_kwargs)

    if 'max_tokens' not in api_kwargs and 'max_completion_tokens' not in api_kwargs:
        api_kwargs['max_tokens'] = max_tokens

    if 'stream' in api_kwargs:
        stream = bool(api_kwargs['stream'])
        del api_kwargs['stream']
    else:
        # Stream by default so the timeout bounds chunk gaps, not total generation time
        stream = True

    actual_api_kwargs = {key: value for key, value in api_kwargs.items() if value is not None}

    # LiveBench provider names → LiteLLM prefixes ('openai/responses' = Responses-API bridge)
    litellm_provider_aliases = {'google': 'gemini', 'together': 'together_ai', 'openai_responses': 'openai/responses'}

    if api_dict is not None and api_dict.get('api_base'):
        litellm_model = f"openai/{model}"
        actual_api_kwargs['api_base'] = api_dict['api_base']
        if api_dict.get('api_key'):
            actual_api_kwargs['api_key'] = api_dict['api_key']
    elif provider and provider not in ('local', ''):
        litellm_model = f"{litellm_provider_aliases.get(provider, provider)}/{model}"
    else:
        litellm_model = model

    # Provider-native params must be listed in allowed_openai_params for
    # LiteLLM to forward them to the API untouched.
    def passthrough(*params: str) -> None:
        present = [p for p in params if p in actual_api_kwargs]
        if present:
            existing = actual_api_kwargs.get('allowed_openai_params') or []
            actual_api_kwargs['allowed_openai_params'] = list(existing) + present

    if 'deepseek' in litellm_model:
        # DeepSeek's API rejects the system role
        messages = [{**m, 'role': 'user'} if m.get('role') == 'system' else m for m in messages]

    if 'anthropic' in litellm_model:
        if 'betas' in actual_api_kwargs:
            actual_api_kwargs['extra_headers'] = {'anthropic-beta': ','.join(actual_api_kwargs.pop('betas'))}
        if 'extra_body' in actual_api_kwargs:
            actual_api_kwargs = actual_api_kwargs | actual_api_kwargs.pop('extra_body')
        if 'thinking' not in actual_api_kwargs:
            actual_api_kwargs['thinking'] = {'type': 'disabled'}
        elif actual_api_kwargs['thinking'].get('type') == 'auto':
            # 'auto' is the beta-client spelling; the standard endpoint only accepts 'adaptive'
            actual_api_kwargs['thinking'] = {**actual_api_kwargs['thinking'], 'type': 'adaptive'}
        passthrough('thinking', 'output_config')
    else:
        passthrough('reasoning_effort')
        if litellm_model.startswith('gemini/'):
            passthrough('thinking_config')
        elif litellm_model.startswith('openai/responses/'):
            passthrough('reasoning')

    if stream and 'stream_options' not in actual_api_kwargs:
        actual_api_kwargs['stream_options'] = {'include_usage': True}

    timeout = actual_api_kwargs.pop('timeout', 600)

    try:
        response = litellm.completion(
            model=litellm_model, messages=messages, stream=stream, timeout=timeout, **actual_api_kwargs
        )
    except Exception as e:
        # Fall back to non-streaming if the provider rejects streaming
        err = str(e).lower()
        if stream and 'stream' in err and any(s in err for s in ('support', 'invalid', 'not allowed', 'verified')):
            logger.warning(f"{litellm_model} rejected streaming, retrying non-streaming: {e}")
            stream = False
            actual_api_kwargs.pop('stream_options', None)
            response = litellm.completion(
                model=litellm_model, messages=messages, stream=False, timeout=timeout, **actual_api_kwargs
            )
        else:
            raise

    if stream:
        chunks = list(response)
        last_usage = None
        for chunk in reversed(chunks):
            if hasattr(chunk, 'usage') and chunk.usage is not None:
                last_usage = chunk.usage
                break
        response = litellm.stream_chunk_builder(chunks, messages=messages)
        if response.usage is not None and last_usage is not None:
            if not response.usage.prompt_tokens and getattr(last_usage, 'prompt_tokens', None):
                response.usage.prompt_tokens = last_usage.prompt_tokens

    message = response.choices[0].message.content
    finish_reason = response.choices[0].finish_reason
    reasoning_content = getattr(response.choices[0].message, 'reasoning_content', None)

    num_tokens = None
    input_tokens = None
    cached_tokens = None
    token_exhaustion = False
    if response.usage is not None:
        num_tokens = response.usage.completion_tokens
        input_tokens = response.usage.prompt_tokens
        cached_tokens = getattr(response.usage, 'cache_read_input_tokens', None) or 0
        if not cached_tokens and hasattr(response.usage, 'prompt_tokens_details') and response.usage.prompt_tokens_details is not None:
            cached_tokens = response.usage.prompt_tokens_details.cached_tokens or 0

    if message is None or message == '':
        if finish_reason == 'length' and reasoning_content:
            logger.warning(f"Token exhaustion: model used all tokens on reasoning ({num_tokens} tokens), no content produced")
            message = ""
            token_exhaustion = True
        else:
            raise Exception(f"No message returned from litellm (finish_reason={finish_reason})")
    if num_tokens is None:
        num_tokens = -1

    metadata: dict[str, Any] | None = None
    if input_tokens is not None:
        metadata = {'input_tokens': input_tokens}
        if cached_tokens is not None:
            metadata['cached_tokens'] = cached_tokens

    if token_exhaustion:
        if metadata is None:
            metadata = {}
        metadata['eval_status'] = 'token_exhaustion'

    return message, num_tokens, metadata


def get_api_function(provider_name: str, use_litellm: bool = False) -> ModelAPI:
    if use_litellm:
        from functools import partial
        return partial(chat_completion_litellm, provider=provider_name)
    if provider_name == 'openai':
        return chat_completion_openai
    elif provider_name == 'openai_responses':
        return chat_completion_openai_responses
    elif provider_name == 'anthropic':
        return chat_completion_anthropic
    elif provider_name == 'mistral':
        return chat_completion_mistral
    elif provider_name == 'together':
        return chat_completion_together
    elif provider_name == 'google':
        return chat_completion_google_generativeai
    elif provider_name == 'aws':
        return chat_completion_aws
    elif provider_name == 'deepinfra':
        return chat_completion_deepinfra
    else:
        return chat_completion_openai
