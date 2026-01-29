import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

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
    preserve_reasoning: bool | None = None

class LitellmModel:
    def __init__(self, **kwargs):
        if 'https' in kwargs['model_name']:
            base_url = cast(str, kwargs['model_name']).rsplit('/', 1)[0]
            kwargs['model_name'] = kwargs['model_name'].replace('https://', 'openai/')
            kwargs['model_kwargs']['api_base'] = base_url
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
                KeyboardInterrupt,
            )
        ),
        reraise=True,
    )
    def _query_completion_generativeai(self, messages: list[dict[str, str]], **kwargs):
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        actual_messages: list[types.ContentOrDict] = []
        system = None
        for message in messages:
            if message['role'] == 'system':
                if system is None:
                    system = types.Content(role='system', parts=[types.Part.from_text(text=message['content'])])
                else:
                    system.parts.append(types.Part.from_text(text=message['content']))
            elif message['role'] == 'user':
                actual_messages.append(types.Content(role='user', parts=[types.Part.from_text(text=message['content'])]))
            elif message['role'] == 'assistant':
                message['role'] = 'model'
                actual_messages.append(message)
            elif message['role'] == 'model':
                actual_messages.append(message)

        safety_settings = [
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY, threshold=types.HarmBlockThreshold.BLOCK_NONE),
        ]

        api_kwargs = self.config.model_kwargs | kwargs

        config = types.GenerateContentConfig(
            safety_settings=safety_settings,
            system_instruction=system,
            **api_kwargs
        )

        actual_model_name = self.config.model_name.split('/')[-1]

        response = client.models.generate_content(model=actual_model_name, contents=actual_messages, config=config)
        
        if response.text is None or response.candidates is None or len(response.candidates) == 0:
            raise Exception("No response returned from Google")
        
        message = response.text

        result: dict[str, Any] = {
            'response': response,
            'content': message,
            'message': response.candidates[0].content,
        }

        if response.usage_metadata is not None:
            if response.usage_metadata.prompt_token_count is not None:
                result['input_tokens'] = response.usage_metadata.prompt_token_count
            if response.usage_metadata.candidates_token_count is not None:
                result['output_tokens'] = response.usage_metadata.candidates_token_count
            if response.usage_metadata.thoughts_token_count is not None:
                result['output_tokens'] += response.usage_metadata.thoughts_token_count

        return result

    @retry(
        stop=stop_after_attempt(15),
        wait=wait_exponential(multiplier=2, min=4, max=120),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(
            (
                KeyboardInterrupt,
            )
        ),
        reraise=True,
    )
    def _query_completion_anthropic_direct(self, messages: list[dict[str, str]], **kwargs):
        """Direct Anthropic SDK call to support features not yet in LiteLLM (e.g., thinking.type: auto)"""
        from anthropic import NOT_GIVEN, Anthropic
        
        # print(f"[DEBUG] _query_completion_anthropic_direct called for model: {self.config.model_name}")
        
        client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        
        api_kwargs = self.config.model_kwargs | kwargs
        actual_api_kwargs = {key: (value if value is not None else NOT_GIVEN) for key, value in api_kwargs.items()}
        
        # Extract system message and sanitize other messages
        system_content = None
        actual_messages = []
        for msg in messages:
            if msg['role'] == 'system':
                if system_content is None:
                    system_content = msg['content']
                else:
                    system_content += "\n" + msg['content']
            else:
                # Sanitize message content to remove any extra fields the API doesn't accept
                sanitized_msg = {"role": msg['role']}
                msg_content = msg.get('content')
                
                if isinstance(msg_content, str):
                    # Simple string content
                    sanitized_msg['content'] = msg_content
                elif isinstance(msg_content, list):
                    # Content is a list of blocks - sanitize each block
                    sanitized_blocks = []
                    for block in msg_content:
                        if isinstance(block, dict):
                            block_type = block.get('type')
                            if block_type == 'text':
                                # Only keep 'type' and 'text' fields
                                sanitized_blocks.append({
                                    "type": "text",
                                    "text": block.get('text', '')
                                })
                            elif block_type == 'thinking':
                                # Keep 'type', 'thinking', and 'signature'
                                thinking_block = {
                                    "type": "thinking",
                                    "thinking": block.get('thinking', '')
                                }
                                if 'signature' in block:
                                    thinking_block['signature'] = block['signature']
                                sanitized_blocks.append(thinking_block)
                            elif block_type == 'redacted_thinking':
                                sanitized_blocks.append({
                                    "type": "redacted_thinking",
                                    "data": block.get('data', '')
                                })
                            else:
                                # Unknown block type - pass through but log
                                # print(f"[DEBUG] Unknown block type in message: {block_type}")
                                sanitized_blocks.append(block)
                        else:
                            sanitized_blocks.append(block)
                    sanitized_msg['content'] = sanitized_blocks
                else:
                    # Unknown content format - pass through
                    sanitized_msg['content'] = msg_content
                
                actual_messages.append(sanitized_msg)
        
        # print(f"[DEBUG] Processed {len(actual_messages)} messages (excluding system)")
        
        # Get the actual model name (remove 'anthropic/' prefix if present)
        actual_model_name = self.config.model_name
        if actual_model_name.startswith('anthropic/'):
            actual_model_name = actual_model_name[len('anthropic/'):]
        
        # Extract special parameters
        betas = actual_api_kwargs.pop('betas', [])
        output_config = actual_api_kwargs.pop('output_config', None)
        thinking = actual_api_kwargs.pop('thinking', None)
        max_tokens = actual_api_kwargs.pop('max_tokens', 8192)
        temperature = actual_api_kwargs.pop('temperature', NOT_GIVEN)
        
        # print(f"[DEBUG] Model: {actual_model_name}, Betas: {betas}")
        # print(f"[DEBUG] thinking config: {thinking}")
        # print(f"[DEBUG] output_config: {output_config}")
        # print(f"[DEBUG] max_tokens: {max_tokens}")
        # print(f"[DEBUG] temperature: {temperature}")
        # print(f"[DEBUG] remaining kwargs: {list(actual_api_kwargs.keys())}")
        
        # Build call kwargs - only include supported parameters
        call_kwargs: dict[str, Any] = {
            'model': actual_model_name,
            'messages': actual_messages,
            'max_tokens': max_tokens,
        }
        
        if system_content:
            call_kwargs['system'] = system_content
        if thinking is not None:
            call_kwargs['thinking'] = thinking
        if temperature is not NOT_GIVEN:
            call_kwargs['temperature'] = temperature
        if betas:
            call_kwargs['betas'] = betas
        
        # Try passing output_config directly first; if that fails, extra_body is the fallback
        # User's working code passes output_config directly to client.beta.messages.stream()
        if output_config is not None:
            call_kwargs['output_config'] = output_config
            # print(f"[DEBUG] Added output_config directly: {output_config}")
        
        # print(f"[DEBUG] Final call_kwargs keys: {list(call_kwargs.keys())}")
        # print(f"[DEBUG] Full call_kwargs: {call_kwargs}")
        
        # Always use beta client for this method (it handles thinking.type=auto)
        # print(f"[DEBUG] client type: {type(client)}")
        # print(f"[DEBUG] client.beta type: {type(client.beta)}")
        # print(f"[DEBUG] client.beta.messages type: {type(client.beta.messages)}")
        
        # Stream the response - matching exact working pattern
        # print(f"[DEBUG] Starting stream request...")
        with client.beta.messages.stream(**call_kwargs) as stream:
            response = stream.get_final_message()
        
        # print(f"[DEBUG] Got response, id: {response.id}")
        # print(f"[DEBUG] Response content blocks: {len(response.content)}")
        
        # Extract non-empty text and build sanitized content for history
        content = ""
        sanitized_content = []
        for i, block in enumerate(response.content):
            block_type = getattr(block, 'type', 'unknown')
            # print(f"[DEBUG]   Block {i}: type={block_type}")
            
            if block_type == "text":
                text = getattr(block, 'text', '')
                # print(f"[DEBUG]     text length: {len(text)}, stripped: {len(text.strip())}")
                # Only include non-empty text blocks in sanitized content
                if text.strip():
                    sanitized_content.append({"type": "text", "text": text})
                    if not content:  # Use first non-empty text as the content
                        content = text.strip()
                        # print(f"[DEBUG]     Using this block as content (first 100 chars): {repr(content[:100])}")
            elif block_type == "thinking":
                thinking_text = getattr(block, 'thinking', '')
                signature = getattr(block, 'signature', '')
                # print(f"[DEBUG]     thinking length: {len(thinking_text)}")
                # Include thinking blocks in sanitized content (needed for multi-turn)
                thinking_block = {"type": "thinking", "thinking": thinking_text}
                if signature:
                    thinking_block["signature"] = signature
                sanitized_content.append(thinking_block)
            elif block_type == "redacted_thinking":
                # Include redacted thinking blocks
                data = getattr(block, 'data', '')
                sanitized_content.append({"type": "redacted_thinking", "data": data})
        
        if not content:
            raise Exception("No text response from Anthropic")
        
        # Create sanitized message for conversation history
        sanitized_message = {
            "role": "assistant",
            "content": sanitized_content
        }
        # print(f"[DEBUG] Sanitized message content blocks: {len(sanitized_content)}")
        
        input_tokens = response.usage.input_tokens if response.usage else 0
        output_tokens = response.usage.output_tokens if response.usage else 0
        # print(f"[DEBUG] Tokens - input: {input_tokens}, output: {output_tokens}")
        
        result: dict[str, Any] = {
            'response': response,
            'content': content,
            'message': sanitized_message,  # Use sanitized message, not full response
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
        }
        
        # print(f"[DEBUG] _query_completion_anthropic_direct completed successfully")
        return result

    def _needs_direct_anthropic_call(self) -> bool:
        """Check if we need to bypass LiteLLM for direct Anthropic SDK call"""
        if 'anthropic' not in self.config.model_name:
            return False
        thinking = self.config.model_kwargs.get('thinking', {})
        return thinking.get('type') == 'auto'

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
        reraise=True,
    )
    def _query_completion(self, messages: list[dict[str, str]], **kwargs):
        if 'deepseek' in self.config.model_name:
            for message in messages:
                if message['role'] == 'system':
                    message['role'] = 'user'
        actual_kwargs = self.config.model_kwargs | kwargs
        reasoning_param_name = 'thinking' if 'anthropic' in self.config.model_name else 'reasoning_effort'
        if 'allowed_openai_params' in actual_kwargs:
            actual_kwargs['allowed_openai_params'] = actual_kwargs['allowed_openai_params'] + [reasoning_param_name]
        else:
            actual_kwargs['allowed_openai_params'] = [reasoning_param_name]

        if 'anthropic' in self.config.model_name:
            if 'betas' in actual_kwargs:
                actual_kwargs['extra_headers'] = {'anthropic-beta': beta for beta in actual_kwargs['betas']}
                del actual_kwargs['betas']
            if 'extra_body' in actual_kwargs:
                actual_kwargs = actual_kwargs | actual_kwargs['extra_body']
                del actual_kwargs['extra_body']
            if 'thinking' not in actual_kwargs:
                actual_kwargs['thinking'] = {'type': 'disabled'}
        try:
            res = litellm.completion(
                model=self.config.model_name, messages=messages, **actual_kwargs
            )

            if actual_kwargs.get('stream', False):
                chunks = []
                for chunk in res:
                    chunks.append(chunk)
                res = litellm.stream_chunk_builder(chunks, messages=messages)
        except litellm.exceptions.InternalServerError as e:
            if "This model's maximum context length is" in str(e):
                raise litellm.exceptions.ContextWindowExceededError(str(e), model=self.config.model_name, llm_provider=self.config.model_name) from e
            raise e

        if res['choices'][0]['finish_reason'] == 'length' and 'max_tokens' in actual_kwargs and res.usage.completion_tokens < actual_kwargs['max_tokens']:
            raise Exception("Model returned length error but max tokens were not reached")

        content = res['choices'][0]['message']['content']


        result = {
            'response': res,
            'message': res['choices'][0]['message']
        }
        if res and res.choices and len(res.choices) > 0:
            result['content'] = content
            result['input_tokens'] = res.usage.prompt_tokens
            result['output_tokens'] = res.usage.completion_tokens
        return result

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING, exc_info=True),
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
        reraise=True,
    )
    def _query_responses(self, messages: list[dict[str, str]], **kwargs):
        system_messages: list[str] = []
        for message in messages:
            if message.get('role') == 'system':
                system_messages.append(message.get('content', ''))
        system_prompt = system_messages[0] if system_messages else None

        res = litellm.responses(
            model=self.config.model_name, input=messages, instructions=system_prompt, **(self.config.model_kwargs | kwargs),
        )

        output_text = ""

        outputs = []

        for output_item in res.output:

            output_item = output_item.model_dump() if not isinstance(output_item, dict) else output_item

            outputs.append(output_item)
            
            if output_item.get('type') == 'message':
                for content in output_item.get('content', []):
                    if content.get('type') == 'output_text':
                        output_text += content.get('text', '')

        result = {
            'response': res,
            'content': output_text,
            'outputs': outputs,
        }
        if res and res.usage is not None:
            result['input_tokens'] = res.usage.input_tokens if not isinstance(res.usage, dict) else res.usage.get('input_tokens', 0)
            result['output_tokens'] = res.usage.output_tokens if not isinstance(res.usage, dict) else res.usage.get('output_tokens', 0)
        return result

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:

        actual_messages: list[dict[str, str]] = []
        for message in messages:
            message_copy = message.copy()
            if 'extra' in message_copy:
                if message_copy['extra'].get('message') is not None:
                    if 'provider_specific_fields' in message_copy['extra']['message']:
                        message_copy['extra']['message'] = message_copy['extra']['message'] | message_copy['extra']['message']['provider_specific_fields']
                        del message_copy['extra']['message']['provider_specific_fields']
                    actual_messages.append(message_copy['extra']['message'])
                elif message_copy['extra'].get('outputs') is not None:
                    actual_messages.extend(message_copy['extra']['outputs'])
                else:
                    del message_copy['extra']
                    actual_messages.append(message_copy)
            else:
                actual_messages.append(message_copy)

        if 'gemini-3' in self.config.model_name:
            result = self._query_completion_generativeai(actual_messages, **kwargs)
        elif self._needs_direct_anthropic_call():
            # print(f"[DEBUG] Routing to direct Anthropic call (thinking.type=auto detected)")
            result = self._query_completion_anthropic_direct(actual_messages, **kwargs)
        elif self.config.api_type == "completion":
            result = self._query_completion(actual_messages, **kwargs)
        elif self.config.api_type == "responses":
            result = self._query_responses(actual_messages, **kwargs)
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
        except Exception as _:
            # logger.warning(
            #     f"Error calculating cost for model {self.config.model_name}: {e}. "
            #     "Please check the 'Updating the model registry' section in the documentation at "
            #     "https://klieret.short.gy/litellm-model-registry Still stuck? Please open a github issue for help!"
            # )
            pass
        self.n_calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        # Handle message serialization - some methods return Pydantic models, some return dicts
        message_data = None
        if 'message' in result:
            msg = result['message']
            if hasattr(msg, 'model_dump'):
                message_data = msg.model_dump()
            elif isinstance(msg, dict):
                message_data = msg
            else:
                message_data = msg
        
        res = {
            "content": content or "",
            "extra": {
                "response": response.model_dump() if hasattr(response, 'model_dump') else response,
                "message": message_data,
                "outputs": result['outputs'] if 'outputs' in result else None
            },
        }

        return res

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config) | {"n_model_calls": self.n_calls, "model_cost": self.cost, "total_input_tokens": self.input_tokens, "total_output_tokens": self.output_tokens}
