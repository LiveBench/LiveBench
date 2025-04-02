import logging
import os
import sys
import time
import traceback
from typing import TYPE_CHECKING, cast
import httpx

from tenacity import retry, stop_after_attempt, retry_if_exception_type, wait_fixed, wait_incrementing

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)


logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    from livebench.model.models import Model

# API setting constants
API_MAX_RETRY = 3
API_RETRY_SLEEP_MIN = 10
API_RETRY_SLEEP_MAX = 60
API_ERROR_OUTPUT = "$ERROR$"


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
    model: "Model", conv, temperature, max_tokens, api_dict=None, stream=False
) -> tuple[str, int]:
    from livebench.model.models import OpenAIModel
    from openai import NOT_GIVEN, OpenAI

    if api_dict is not None:
        client = OpenAI(
            api_key=api_dict["api_key"], base_url=api_dict["api_base"], timeout=httpx.Timeout(timeout=2400.0, connect=10.0)
        )

    else:
        client = OpenAI(timeout=1000)

    messages = conv.to_openai_api_messages()
    messages = [m for m in messages if m['role'] != 'system']
    if isinstance(model, OpenAIModel) and model.inference_api:
        messages[0]['content'] = 'Formatting reenabled\n' + messages[0]['content']
    try:
        if stream:
            stream = client.chat.completions.create(
                model=model.api_name,
                messages=messages,
                n=1,
                temperature=(
                    temperature
                    if not isinstance(model, OpenAIModel) or not model.inference_api
                    else NOT_GIVEN
                ),
                max_completion_tokens=(
                    max_tokens if isinstance(model, OpenAIModel) and not model.inference_api else NOT_GIVEN
                ),
                max_tokens=(
                    max_tokens if not isinstance(model, OpenAIModel) else NOT_GIVEN
                ),
                reasoning_effort=model.api_kwargs['reasoning_effort'] if model.api_kwargs is not None and 'reasoning_effort' in model.api_kwargs else NOT_GIVEN,
                stream=True,
                stream_options={'include_usage': True}
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
            response = client.chat.completions.create(
                model=model.api_name,
                messages=messages,
                n=1,
                temperature=(
                    temperature
                    if not isinstance(model, OpenAIModel) or not model.inference_api
                    else NOT_GIVEN
                ),
                max_completion_tokens=(
                    max_tokens if isinstance(model, OpenAIModel) and not model.inference_api else NOT_GIVEN
                ),
                max_tokens=(
                    max_tokens if not isinstance(model, OpenAIModel) else NOT_GIVEN
                ),
                reasoning_effort=model.api_kwargs['reasoning_effort'] if model.api_kwargs is not None and 'reasoning_effort' in model.api_kwargs else NOT_GIVEN,
                stream=False,
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
                if hasattr(response.usage, 'reasoning_tokens'):
                    num_tokens += response.usage.reasoning_tokens
            else:
                num_tokens = None

        if message is None or message == '':
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
def chat_completion_openai_responses(model, conv, temperature, max_tokens, api_dict=None) -> tuple[str, int]:
    from livebench.model.models import OpenAIResponsesModel
    from openai import NOT_GIVEN, OpenAI

    if api_dict is not None:
        client = OpenAI(
            api_key=api_dict["api_key"], base_url=api_dict["api_base"], timeout=httpx.Timeout(timeout=2400.0, connect=10.0)
        )
    else:
        client = OpenAI(timeout=2400)

    model = cast(OpenAIResponsesModel, model)

    messages = conv.to_openai_api_messages()
    developer_message_index = None
    for i, message in enumerate(messages):
        if message["role"] == "system" or message["role"] == "developer":
            developer_message_index = i
            break
    developer_message = messages[developer_message_index]['content']
    messages = messages[0:developer_message_index] + messages[developer_message_index+1:]
    developer_message = 'Formatting reenabled\n' + developer_message

    reasoning_effort = model.api_kwargs['reasoning_effort'] if model.api_kwargs is not None and 'reasoning_effort' in model.api_kwargs else NOT_GIVEN
    max_output_tokens = max_tokens if not model.inference_api else NOT_GIVEN
    temperature = temperature if not model.inference_api else NOT_GIVEN

    input = '\n'.join([message['content'] for message in messages])

    output_text = ""
    output_tokens = None

    stream = client.responses.create(
        model=model.api_name,
        instructions=developer_message,
        input=input,
        reasoning={"effort": reasoning_effort} if reasoning_effort is not NOT_GIVEN else NOT_GIVEN,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        stream = True
    )

    for event in stream:
        print(event)
        if event.type == "response.completed":
            output_tokens = event.response.usage.output_tokens
        elif event.type == "response.output_text.delta":
            output_text += event.delta
        elif event.type == "response.failed":
            raise Exception(f"Failed to generate response ({event.error.code}): {event.error.message}")
        elif event.type == "response.incomplete":
            raise Exception("Response was cut short: " + event.incomplete_details.reason)

    print(output_text)
    print(output_tokens)

    return output_text, output_tokens

@retry(
    stop=stop_after_attempt(API_MAX_RETRY),
    wait=wait_fixed(API_RETRY_SLEEP_MIN),
    retry=retry_if_exception_type(Exception),
    after=retry_log,
    retry_error_callback=retry_fail
)
def chat_completion_aws(model, conv, temperature, max_tokens, api_dict=None) -> tuple[str, int]:
    import boto3

    brt = boto3.client("bedrock-runtime", region_name="us-east-1")
    prompt = [text for role, text in conv.messages if role == "user"][0]

    response = brt.converse(
        modelId=model.api_name,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={
            "maxTokens": max_tokens,
            "temperature": temperature,
        },
    )

    output = response["output"]["message"]["content"][0]["text"]
    num_tokens = response["usage"]["outputTokens"]

    return output, num_tokens


@retry(
    stop=stop_after_attempt(API_MAX_RETRY),
    wait=wait_fixed(API_RETRY_SLEEP_MIN),
    retry=retry_if_exception_type(Exception),
    after=retry_log,
    retry_error_callback=retry_fail
)
def chat_completion_deepseek(model, conv, temperature, max_tokens, api_dict=None) -> tuple[str, int]:
    if api_dict is not None and "api_key" in api_dict:
        api_key = api_dict["api_key"]
    else:
        api_key = os.environ["DEEPSEEK_API_KEY"]

    from livebench.model.models import DeepseekModel
    model = cast(DeepseekModel, model)

    print("sleeping for 3 sec")
    time.sleep(3)
    from openai import NOT_GIVEN, OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    messages = conv.to_openai_api_messages()

    kwargs = {
        'temperature': temperature if not model.reasoner else NOT_GIVEN,
        'max_tokens': max_tokens
    }
    kwargs.update(model.api_kwargs)

    response = client.chat.completions.create(
        model=model.api_name,
        messages=messages,
        n=1,
        stream=False,
        **kwargs
    )
    message = response.choices[0].message.content
    if message is None:
        raise Exception("No message returned from DeepSeek")
    output = message
    num_tokens = response.usage.completion_tokens
    if hasattr(response.usage, 'reasoning_tokens'):
        num_tokens += response.usage.reasoning_tokens

    return output, num_tokens


@retry(
    stop=stop_after_attempt(API_MAX_RETRY),
    wait=wait_fixed(API_RETRY_SLEEP_MIN),
    retry=retry_if_exception_type(Exception),
    after=retry_log,
    retry_error_callback=retry_fail
)
def chat_completion_nvidia(model, conv, temperature, max_tokens, api_dict=None) -> tuple[str, int]:
    if api_dict is not None and "api_key" in api_dict:
        api_key = api_dict["api_key"]
    else:
        api_key = os.environ["NVIDIA_API_KEY"]

    print("sleeping for 2 sec")
    time.sleep(2)

    from openai import OpenAI

    client = OpenAI(
        api_key=api_key, base_url="https://integrate.api.nvidia.com/v1"
    )
    messages = conv.to_openai_api_messages()
    response = client.chat.completions.create(
        model=model.api_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        stream=False,
    )
    message = response.choices[0].message.content
    if message is None:
        raise Exception("No message returned from NVIDIA")

    return message, response.usage.completion_tokens


@retry(
    stop=stop_after_attempt(API_MAX_RETRY),
    wait=wait_fixed(API_RETRY_SLEEP_MIN),
    retry=retry_if_exception_type(Exception),
    after=retry_log,
    retry_error_callback=retry_fail
)
def chat_completion_xai(model, conv, temperature, max_tokens, api_dict=None) -> tuple[str, int]:
    if api_dict is not None and "api_key" in api_dict:
        api_key = api_dict["api_key"]
    else:
        api_key = os.environ["XAI_API_KEY"]

    from openai import OpenAI

    client = OpenAI(base_url="https://api.x.ai/v1", api_key=api_key)
    messages = conv.to_openai_api_messages()

    response = client.chat.completions.create(
        model=model.api_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,
        stream=False,
    )
    message = response.choices[0].message.content
    if message is None:
        raise Exception("No message returned from X.AI")
    if response.usage is not None:
        num_tokens = response.usage.completion_tokens
        if hasattr(response.usage, 'reasoning_tokens'):
            num_tokens += response.usage.reasoning_tokens
    else:
        num_tokens = None

    return message, num_tokens


@retry(
    stop=stop_after_attempt(API_MAX_RETRY),
    wait=wait_fixed(API_RETRY_SLEEP_MIN),
    retry=retry_if_exception_type(Exception),
    after=retry_log,
    retry_error_callback=retry_fail
)
def chat_completion_vertex(
    model, conv, temperature, max_tokens, api_dict=None, project_name="DEFAULT"
) -> tuple[str, int]:
    import vertexai
    from vertexai.preview.generative_models import GenerativeModel

    print("sleeping for 5 sec")
    time.sleep(5)
    vertexai.init(project=project_name, location="us-central1")
    generative_multimodal_model = GenerativeModel(model.api_name)
    prompt = [text for role, text in conv.messages if role == "user"][0]
    response = generative_multimodal_model.generate_content([prompt])
    message = response.candidates[0].content.parts[0].text
    if message is None:
        raise Exception("No message returned from Vertex")

    return message.strip(), response.usage.output_tokens

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
    model, conv, temperature, max_tokens, api_dict=None
) -> tuple[str, int]:
    from google import genai
    from google.genai import types

    if api_dict is not None and "api_key" in api_dict:
        api_key = api_dict["api_key"]
    else:
        api_key = os.environ["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})

    safety_settings = [
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
    ]

    system = [text for role, text in conv.messages if role == "system"][0] if len([text for role, text in conv.messages if role == "system"]) > 0 else None
    prompt = [text for role, text in conv.messages if role == "user"][0]

    kwargs = {'safety_settings': safety_settings, 'system_instruction': system}

    if model.api_kwargs is not None:
        kwargs.update(model.api_kwargs)

    config = types.GenerateContentConfig(**kwargs)

    response = client.models.generate_content(
        model=model.api_name,
        contents=prompt,
        config=config
    )

    if response is None or response.candidates is None or len(response.candidates) == 0 or response.candidates[0].content is None or response.candidates[0].content.parts is None or len(response.candidates[0].content.parts) == 0:
        msg = "No message returned from Google Generative AI"
        if response is not None and response.candidates is not None and len(response.candidates) > 0 and response.candidates[0].finish_reason is not None:
            msg += f" - finish reason: {response.candidates[0].finish_reason}"
        raise Exception(msg)

    if len(response.candidates[0].content.parts) > 1:
        # if using a reasoning model, don't return the CoT text
        final_res = ''
        cot = ''
        for part in response.candidates[0].content.parts:
            if part.thought:
                cot += part.text
            else:
                final_res += part.text
    else:
        final_res = response.candidates[0].content.parts[0].text
        cot = ''

    full = final_res + '\n' + cot

    tokens = client.models.count_tokens(
        model=model.api_name,
        contents=full
    ).total_tokens

    return final_res, tokens


@retry(
    stop=stop_after_attempt(API_MAX_RETRY),
    wait=wait_fixed(API_RETRY_SLEEP_MIN),
    retry=retry_if_exception_type(Exception),
    after=retry_log,
    retry_error_callback=retry_fail
)
def chat_completion_together(model, conv, temperature, max_tokens, api_dict=None) -> tuple[str, int]:
    from together import Together

    if api_dict is not None and "api_key" in api_dict:
        api_key = api_dict["api_key"]
    else:
        api_key = os.environ["TOGETHER_API_KEY"]
    client = Together(api_key=api_key)

    messages = conv.to_openai_api_messages()
    
    messages = [message for message in messages if message['role'] == 'user']

    kwargs = {'max_tokens': max_tokens, 'temperature': temperature}
    if model.api_kwargs is not None:
        kwargs.update(model.api_kwargs)

    response = client.chat.completions.create(
        model=model.api_name,
        messages=messages,
        **kwargs
    )

    return response.choices[0].message.content, response.usage.completion_tokens


@retry(
    stop=stop_after_attempt(API_MAX_RETRY),
    wait=wait_fixed(API_RETRY_SLEEP_MIN),
    retry=retry_if_exception_type(Exception),
    after=retry_log,
    retry_error_callback=retry_fail
)
def chat_completion_perplexity(model, conv, temperature, max_tokens, api_dict=None) -> tuple[str, int]:
    if api_dict is not None and "api_key" in api_dict:
        api_key = api_dict["api_key"]
    else:
        api_key = os.environ["PERPLEXITY_API_KEY"]

    from openai import OpenAI

    client = OpenAI(base_url="https://api.perplexity.ai", api_key=api_key)
    messages = conv.to_openai_api_messages()

    response = client.chat.completions.create(
        model=model.api_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content, response.usage.completion_tokens


@retry(
    stop=stop_after_attempt(API_MAX_RETRY),
    wait=wait_fixed(API_RETRY_SLEEP_MIN),
    retry=retry_if_exception_type(Exception),
    after=retry_log,
    retry_error_callback=retry_fail
)
def chat_completion_openai_azure(model, conv, temperature, max_tokens, api_dict=None) -> tuple[str, int]:
    import openai
    openai.api_type = "azure"
    openai.api_version = "2023-07-01-preview"
    if api_dict is not None:
        openai.base_url = api_dict["api_base"]
        openai.api_key = api_dict["api_key"]
    else:
        openai.base_url = os.environ["AZURE_OPENAI_ENDPOINT"]
        openai.api_key = os.environ["AZURE_OPENAI_KEY"]

    if "azure-" in model:
        model = model[6:]

    from openai import OpenAI
    client = OpenAI()

    messages = conv.to_openai_api_messages()
    response = client.chat.completions.create(
        engine=model.api_name,
        messages=messages,
        n=1,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    output = response.choices[0].message.content
    if output is None:
        raise Exception("No message returned from Azure OpenAI")

    return output, response.usage.completion_tokens


@retry(
    stop=stop_after_attempt(API_MAX_RETRY),
    wait=wait_fixed(API_RETRY_SLEEP_MIN),
    retry=retry_if_exception_type(Exception),
    after=retry_log,
    retry_error_callback=retry_fail
)
def chat_completion_anthropic(model, conv, temperature, max_tokens, api_dict=None) -> tuple[str, int]:
    if api_dict is not None and "api_key" in api_dict:
        api_key = api_dict["api_key"]
    else:
        api_key = os.environ["ANTHROPIC_API_KEY"]

    from anthropic import NOT_GIVEN, Anthropic
    c = Anthropic(api_key=api_key)
    prompt = [text for role, text in conv.messages if role == "Human"][0]

    kwargs = {
        'max_tokens': max_tokens,
        'temperature': temperature
    }

    if model.api_kwargs is not None:
        kwargs.update(model.api_kwargs)

    for k in kwargs:
        if kwargs[k] == None:
            kwargs[k] = NOT_GIVEN

    message = []

    with c.messages.stream(
        model=model.api_name,
        messages=[
            {"role": "user", "content": prompt}
        ],
        **kwargs
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

    del kwargs['max_tokens']
    del kwargs['temperature']

    try:
        tokens = c.messages.count_tokens(
            model=model.api_name,
            messages=[
                {"role": "assistant", "content": message}
            ],
            **kwargs
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
def chat_completion_mistral(model, conv, temperature, max_tokens, api_dict=None) -> tuple[str, int]:
    if api_dict is not None and "api_key" in api_dict:
        api_key = api_dict["api_key"]
    else:
        api_key = os.environ["MISTRAL_API_KEY"]

    from mistralai import Mistral
    client = Mistral(api_key=api_key)

    messages = conv.to_openai_api_messages()
    chat_response = client.chat.complete(
        model=model.api_name,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=messages,
    )

    if chat_response.choices[0].message.content is None:
        raise Exception("No message returned from Mistral")

    return chat_response.choices[0].message.content.strip(), chat_response.usage.completion_tokens


@retry(
    stop=stop_after_attempt(API_MAX_RETRY),
    wait=wait_fixed(API_RETRY_SLEEP_MIN),
    retry=retry_if_exception_type(Exception),
    after=retry_log,
    retry_error_callback=retry_fail
)
def chat_completion_cohere(model, conv, temperature, max_tokens, api_dict=None) -> tuple[str, int]:
    if api_dict is not None and "api_key" in api_dict:
        api_key = api_dict["api_key"]
    else:
        api_key = os.environ["CO_API_KEY"]

    import cohere
    co = cohere.Client(api_key=api_key)
    prompt = [text for role, text in conv.messages if role == "user"][0]

    response = co.chat(
        model=model.api_name,
        max_tokens=min(max_tokens, 4000),
        temperature=temperature,
        message=prompt,
    )
    if response.text is None:
        raise Exception("No message returned from Cohere")

    return response.text.strip(), response.usage.output_tokens


@retry(
    stop=stop_after_attempt(API_MAX_RETRY),
    wait=wait_fixed(API_RETRY_SLEEP_MIN),
    retry=retry_if_exception_type(Exception),
    after=retry_log,
    retry_error_callback=retry_fail
)
def chat_completion_palm(chat_state, model, conv, temperature, max_tokens) -> tuple[str, int]:
    from fastchat.serve.api_provider import init_palm_chat

    assert model.api_name == "palm-2-chat-bison-001"

    if chat_state is None:
        chat_state = init_palm_chat("chat-bison@001")

    parameters = {
        "temperature": temperature,
        "top_p": 0.8,
        "top_k": 40,
        "max_output_tokens": max_tokens,
    }

    response = chat_state.send_message(conv.messages[-2][1], **parameters)
    if response.text is None:
        raise Exception("No message returned from PaLM")

    return chat_state, response.text, response.usage.output_tokens
