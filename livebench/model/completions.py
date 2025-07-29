import logging
import os
import sys
import time
import traceback
from typing import Protocol, cast
import httpx

from openai import Stream
from openai.types.chat import ChatCompletionChunk, ChatCompletion
from tenacity import retry, stop_after_attempt, retry_if_exception_type, wait_fixed, wait_incrementing

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)


logger = logging.getLogger(__name__)

# API setting constants
API_MAX_RETRY = 3
API_RETRY_SLEEP_MIN = 10
API_RETRY_SLEEP_MAX = 60
API_ERROR_OUTPUT = "$ERROR$"

# model api function takes in Model, list of messages, temperature, max tokens, api kwargs, and an api dict
# returns tuple of (output, num tokens)
Conversation = list[dict[str, str]]
API_Kwargs = dict[str, str | float | int | dict[str, str] | None]

class ModelAPI(Protocol):
    def __call__(self, model: str, messages: Conversation, temperature: float, max_tokens: int, model_api_kwargs: API_Kwargs | None, api_dict: dict[str, str] | None, stream: bool) -> tuple[str, int]:
        ...


def retry_fail(_):
    print("all retries failed")
    return API_ERROR_OUTPUT, 0


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
) -> tuple[str, int]:
    from openai import NOT_GIVEN, OpenAI

    if api_dict is not None:
        client = OpenAI(
            api_key=api_dict["api_key"], base_url=api_dict["api_base"], timeout=httpx.Timeout(timeout=2400.0, connect=10.0)
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
            try:
                for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
                        message += chunk.choices[0].delta.content
                    if chunk.usage is not None:
                        num_tokens = chunk.usage.completion_tokens
                        if hasattr(chunk.usage, 'reasoning_tokens'):
                            num_tokens += chunk.usage.reasoning_tokens
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
            if response.usage is not None:
                num_tokens = response.usage.completion_tokens
                if hasattr(response.usage, 'completion_tokens_details') and hasattr(response.usage.completion_tokens_details, 'reasoning_tokens'):
                    reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens
                    if num_tokens is not None and reasoning_tokens is not None:
                        num_tokens += reasoning_tokens
            else:
                num_tokens = None

        if message is None or message == '':
            print(response)
            raise Exception("No message returned from OpenAI")
        if num_tokens is None:
            num_tokens = -1
        output = message

        return output, num_tokens
    except Exception as e:
        if "invalid_prompt" in str(e).lower():
            print("invalid prompt (model refusal), giving up")
            return API_ERROR_OUTPUT, 0
        raise e

@retry(
    stop=stop_after_attempt(1),
    wait=wait_fixed(API_RETRY_SLEEP_MIN),
    retry=retry_if_exception_type(Exception),
    after=retry_log,
    retry_error_callback=retry_fail
)
def chat_completion_openai_responses(model: str, messages: Conversation, temperature: float, max_tokens: int, model_api_kwargs: API_Kwargs | None = None, api_dict: dict[str, str] | None = None, stream: bool = False) -> tuple[str, int]:
    from openai import NOT_GIVEN, OpenAI

    if api_dict is not None:
        client = OpenAI(
            api_key=api_dict["api_key"], base_url=api_dict["api_base"], timeout=httpx.Timeout(timeout=2400.0, connect=10.0)
        )
    else:
        client = OpenAI(timeout=2400)

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
) -> tuple[str, int]:
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
    if response.usage_metadata is not None:
        num_tokens = response.usage_metadata.candidates_token_count
        if response.usage_metadata.thoughts_token_count is not None:
            num_tokens += response.usage_metadata.thoughts_token_count
    
    if num_tokens is None:
        num_tokens = -1
    
    
    return message, num_tokens


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

    with c.messages.stream(
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

    try:
        tokens = c.messages.count_tokens(
            model=model,
            messages=[
                {"role": "assistant", "content": message}
            ],
            **actual_api_kwargs
        ).input_tokens
    except Exception as e:
        print('Failed to count tokens:', e)
        traceback.print_exc()
        tokens = -1

    text_messages = [c for c in message if c['type'] == 'text']
    if len(text_messages) == 0:
        raise Exception("No response from Anthropic")
    message_text = text_messages[0]['text']

    return message_text, tokens


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

    from mistralai import Mistral, UNSET
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

    return response


def chat_completion_deepinfra(model: str, messages: Conversation, temperature: float, max_tokens: int, model_api_kwargs: API_Kwargs | None = None, api_dict: dict[str, str] | None = None, stream: bool = False) -> tuple[str, int]:
    from openai import OpenAI, NOT_GIVEN

    if api_dict is not None and "api_key" in api_dict:
        api_key = api_dict["api_key"]
    else:
        api_key = os.environ["DEEPINFRA_API_KEY"]

    client = OpenAI(api_key=api_key, base_url="https://api.deepinfra.com/v1/openai", timeout=httpx.Timeout(timeout=2400.0, connect=10.0))

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

def get_api_function(provider_name: str) -> ModelAPI:
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
