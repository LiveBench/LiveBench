import copy
import json
import logging
import os
import time
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


class LengthFinishReasonError(Exception):
    """Raised when the model returns finish_reason='length' but completion_tokens < max_tokens."""


_DEFAULT_MAX_ATTEMPTS = 15
LENGTH_FINISH_REASON_MAX_ATTEMPTS = 3


def _length_aware_stop(retry_state) -> bool:
    """tenacity stop: cap LengthFinishReasonError retries, retry everything else normally."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if isinstance(exc, LengthFinishReasonError):
        return retry_state.attempt_number >= LENGTH_FINISH_REASON_MAX_ATTEMPTS
    return retry_state.attempt_number >= _DEFAULT_MAX_ATTEMPTS


def _cached_tokens(details) -> int:
    """Cached-token count from a token-details object (0 if absent)."""
    if details is None:
        return 0
    return getattr(details, 'cached_tokens', 0) or 0


def _set_cache_breakpoint(message: dict) -> None:
    """Mark a message's last text block with an ephemeral cache_control breakpoint.

    Mutates in place, so callers must pass a copy, never the agent's own history.
    """
    ephemeral = {'type': 'ephemeral'}
    content = message.get('content')
    if isinstance(content, str):
        if content.strip():
            message['content'] = [{'type': 'text', 'text': content, 'cache_control': ephemeral}]
    elif isinstance(content, list) and content:
        target = None
        for block in reversed(content):
            if isinstance(block, dict) and block.get('type') == 'text':
                target = block
                break
        if target is None and isinstance(content[-1], dict):
            target = content[-1]
        if target is not None:
            target['cache_control'] = ephemeral


@dataclass
class LitellmModelConfig:
    model_name: str
    api_type: Literal["completion", "responses"] = "completion"
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    litellm_model_registry: Path | str | None = os.getenv("LITELLM_MODEL_REGISTRY_PATH")
    preserve_reasoning: bool | None = None
    # gemini-3 uses the genai SDK by default; False routes it through litellm (the
    # replay keeps provider_specific_fields nested so thought signatures survive)
    native_gemini: bool = True

class LitellmModel:
    def __init__(self, **kwargs):
        if 'https' in kwargs['model_name']:
            # model_name arrives as "<base-url>/<provider-model-name>". When the caller
            # already supplied api_base, split on it exactly — the provider model name
            # may itself contain slashes (minimax/minimax-m2.7, thinkingmachines/Inkling),
            # which a bare rsplit misparses into the base URL.
            api_base = kwargs.get('model_kwargs', {}).get('api_base')
            if api_base and cast(str, kwargs['model_name']).startswith(api_base):
                model = cast(str, kwargs['model_name'])[len(api_base):].lstrip('/')
                kwargs['model_name'] = f"openai/{model}"
            else:
                base_url = cast(str, kwargs['model_name']).rsplit('/', 1)[0]
                kwargs['model_name'] = kwargs['model_name'].replace('https://', 'openai/')
                kwargs['model_kwargs']['api_base'] = base_url
        self.config = LitellmModelConfig(**kwargs)
        self.cost = 0.0
        self.n_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cached_tokens = 0
        self.cache_creation_tokens = 0
        # Stateful Responses-API chaining: only send the new turn each time
        # (previous_response_id + store) instead of replaying the whole convo.
        # None = untried, False = provider can't chain (stateless API / ZDR org).
        self.previous_response_id = None
        self._responses_sent_upto = 0
        self._responses_chaining: bool | None = None
        # Observability for the empty-response resampling (see _query_completion_anthropic_direct):
        # empty_responses = total empty completions seen; empty_resamples_recovered =
        # turns where a resample turned an initial empty into usable text.
        self.empty_responses = 0
        self.empty_resamples_recovered = 0
        # Consecutive-degrade early-abort: number of turns in a row that degraded to
        # empty despite resampling. Once it reaches MSWEA_EMPTY_DEGRADE_ABORT, the
        # instance is treated as being in a sustained-collapse state and resampling is
        # skipped (fail fast) until the model produces usable text again. This model
        # object is created per-instance (run_batch.get_model in a thread pool), so the
        # counter is naturally per-instance -- no sharing or reset needed.
        self.consecutive_empty_degrades = 0
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

        if response.candidates is None or len(response.candidates) == 0:
            raise Exception(
                "No response returned from Google: no candidates "
                f"(prompt_feedback={getattr(response, 'prompt_feedback', None)})"
            )

        if response.text is None:
            # A candidate came back but with no visible text. Two distinct cases,
            # previously conflated into one blind-retried "No response" exception:
            cand = response.candidates[0]
            finish = str(getattr(cand, 'finish_reason', ''))
            um = response.usage_metadata
            thoughts = getattr(um, 'thoughts_token_count', None) if um is not None else None
            if finish.endswith('MAX_TOKENS'):
                # Thought-only response: the whole output budget went to thinking.
                # That is token exhaustion, not a transient failure -- the outer
                # tenacity retry would reproduce it while backing off up to 120s.
                # Return empty content instead; the agent's format-error nudge
                # advances the loop, bounded by step_limit.
                self.empty_responses += 1
                logger.warning(
                    f"Gemini thought-only response (finish_reason={finish}, "
                    f"thoughts_tokens={thoughts}); treating as token exhaustion with empty content"
                )
                message = ""
            else:
                # Genuinely empty text without MAX_TOKENS: transient Google-side
                # empty candidate. Retrying works -- but say what actually came
                # back so the retry log is diagnosable.
                raise Exception(
                    f"Empty text response from Google (finish_reason={finish}, "
                    f"thoughts_tokens={thoughts}, "
                    f"prompt_feedback={getattr(response, 'prompt_feedback', None)})"
                )
        else:
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
            if response.usage_metadata.cached_content_token_count is not None:
                result['cached_tokens'] = response.usage_metadata.cached_content_token_count

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
                                # Unknown block type - pass through
                                sanitized_blocks.append(block)
                        else:
                            sanitized_blocks.append(block)
                    sanitized_msg['content'] = sanitized_blocks
                else:
                    # Unknown content format - pass through
                    sanitized_msg['content'] = msg_content
                
                actual_messages.append(sanitized_msg)
        
        # Get the actual model name (remove 'anthropic/' prefix if present)
        actual_model_name = self.config.model_name
        if actual_model_name.startswith('anthropic/'):
            actual_model_name = actual_model_name[len('anthropic/'):]
        
        # Extract special parameters
        betas = actual_api_kwargs.pop('betas', [])
        # output_config can be at the top level OR nested inside extra_body
        # (LiteLLM uses extra_body as a pass-through wrapper; the direct SDK
        # doesn't need it, so we flatten it here.)
        extra_body = actual_api_kwargs.pop('extra_body', {}) or {}
        output_config = actual_api_kwargs.pop('output_config', None) or extra_body.get('output_config', None)
        # Server-side refusal fallback (opt-in). SDK 0.105.2 has no typed `fallbacks`
        # param, so it must ride in the request body via extra_body; pair with the
        # server-side-fallback-2026-06-01 beta (set in the model config's `betas`).
        fallbacks = actual_api_kwargs.pop('fallbacks', None) or extra_body.get('fallbacks', None)
        thinking = actual_api_kwargs.pop('thinking', None)
        max_tokens = actual_api_kwargs.pop('max_tokens', 8192)
        temperature = actual_api_kwargs.pop('temperature', NOT_GIVEN)
        
        # Build call kwargs - only include supported parameters
        call_kwargs: dict[str, Any] = {
            'model': actual_model_name,
            'messages': actual_messages,
            'max_tokens': max_tokens,
        }
        
        if system_content:
            call_kwargs['system'] = [{
                "type": "text",
                "text": system_content,
                "cache_control": {"type": "ephemeral"},
            }]
        if actual_messages:
            # Copy the last message first: the helper mutates in place, and sanitized
            # blocks can still alias the caller's history.
            actual_messages[-1] = copy.deepcopy(actual_messages[-1])
            _set_cache_breakpoint(actual_messages[-1])
        if thinking is not None:
            call_kwargs['thinking'] = thinking
        if temperature is not NOT_GIVEN:
            call_kwargs['temperature'] = temperature
        if betas:
            call_kwargs['betas'] = betas
        if output_config is not None:
            call_kwargs['output_config'] = output_config
        if fallbacks:
            call_kwargs['extra_body'] = {'fallbacks': fallbacks}

        # Always use beta client for this method (it handles thinking.type=auto)
        #
        # Bounded resample on empty completions: fable-5 (and adaptive-thinking
        # models generally) intermittently return a turn with NO usable text. Left
        # unhandled, the first empty gets stored as an empty assistant turn + a
        # format-error nudge, which the model then mirrors, collapsing into a
        # self-reinforcing empty loop that burns the whole step budget.
        # Because temperature=1, each retry is an INDEPENDENT sample, so resampling
        # the SAME (still-clean) context recovers most transient empties before any
        # poisoning starts. Tuned by MSWEA_EMPTY_RESPONSE_RETRIES (default 5 -> ON so
        # fable-style empty-completion loops don't silently burn the step budget; set
        # to 0 to disable, i.e. the loop runs exactly once = old behavior). Capped at 8
        # so a misconfig can't loop away. On exhaustion we fall through to the same
        # degrade-to-empty path as before, so the worst case is unchanged.
        max_empty_retries = min(int(os.getenv("MSWEA_EMPTY_RESPONSE_RETRIES", "5")), 8)
        # Early-abort on sustained collapse: if the last N turns all degraded to empty
        # even after full resampling, resampling won't fix it -- stop paying the tax
        # (up to 6 API calls + ~23s backoff) on every doomed step and take a single
        # attempt instead. Gated by MSWEA_EMPTY_DEGRADE_ABORT (default 3; set 0 to
        # disable = always resample, i.e. previous behavior). The counter resets the
        # moment the model returns usable text, so full resampling resumes on recovery.
        degrade_abort = int(os.getenv("MSWEA_EMPTY_DEGRADE_ABORT", "3"))
        if degrade_abort > 0 and self.consecutive_empty_degrades >= degrade_abort:
            max_empty_retries = 0
        content = ""
        sanitized_content = []
        response = None
        for _empty_attempt in range(max_empty_retries + 1):
            with client.beta.messages.stream(**call_kwargs) as stream:
                response = stream.get_final_message()

            # Server-side refusal fallback detection: if a fallback fired, this turn was
            # served by a DIFFERENT model (e.g. claude-opus-4-8). Its thinking blocks carry
            # that model's signature; replaying them to the requested model next turn can be
            # rejected as cross-model/modified thinking. So on a fallback turn keep only the
            # text (drop thinking/redacted_thinking) -- the agent only needs the action text.
            served_model = getattr(response, 'model', None) or actual_model_name
            is_fallback = bool(served_model) and (actual_model_name not in str(served_model))

            # Extract non-empty text and build sanitized content for history
            content = ""
            sanitized_content = []
            for block in response.content:
                block_type = getattr(block, 'type', 'unknown')

                if block_type == "text":
                    text = getattr(block, 'text', '')
                    # Only include non-empty text blocks in sanitized content
                    if text.strip():
                        sanitized_content.append({"type": "text", "text": text})
                        if not content:  # Use first non-empty text as the content
                            content = text.strip()
                elif block_type == "thinking":
                    if is_fallback:
                        continue  # drop cross-model thinking to avoid replay rejection
                    thinking_text = getattr(block, 'thinking', '')
                    signature = getattr(block, 'signature', None)
                    # Include thinking blocks in sanitized content (needed for multi-turn)
                    thinking_block = {"type": "thinking", "thinking": thinking_text}
                    if signature is not None:
                        thinking_block["signature"] = signature
                    sanitized_content.append(thinking_block)
                elif block_type == "redacted_thinking":
                    if is_fallback:
                        continue
                    # Include redacted thinking blocks
                    data = getattr(block, 'data', '')
                    sanitized_content.append({"type": "redacted_thinking", "data": data})
                # a 'fallback' marker block (present on fallback turns) is intentionally skipped

            if content:
                if _empty_attempt > 0:
                    # An earlier attempt this turn was empty; this resample recovered.
                    self.empty_resamples_recovered += 1
                    logger.warning(
                        f"Recovered from empty response after {_empty_attempt} resample(s)"
                    )
                # Usable text this turn -> not (or no longer) in sustained collapse.
                self.consecutive_empty_degrades = 0
                break

            # No usable text on this attempt. Count it (observable in the trajectory
            # via model_stats) and resample if we still have attempts left.
            self.empty_responses += 1
            if _empty_attempt < max_empty_retries:
                backoff = min(2 ** _empty_attempt, 8)
                logger.warning(
                    "Empty text response from Anthropic; resampling "
                    f"(attempt {_empty_attempt + 1}/{max_empty_retries + 1}, sleeping {backoff}s)"
                )
                time.sleep(backoff)

        if not content:
            # The model returned no usable text (e.g. only a thinking block, or a
            # transient empty completion). Don't crash the whole instance: return
            # empty content so the agent loop raises a non-terminating FormatError,
            # nudges the model, and continues (bounded by step_limit). Ensure the
            # stored assistant turn is non-empty so the *next* API call stays valid
            # (Anthropic rejects assistant messages with empty content).
            self.consecutive_empty_degrades += 1
            logger.warning(
                "Empty text response from Anthropic; degrading to empty turn so the agent can retry "
                f"(consecutive degrades: {self.consecutive_empty_degrades})"
            )
            content = ""
            if not sanitized_content:
                sanitized_content = [{"type": "text", "text": "(no response)"}]
        
        # Create sanitized message for conversation history
        sanitized_message = {
            "role": "assistant",
            "content": sanitized_content
        }
        
        input_tokens = response.usage.input_tokens if response.usage else 0
        output_tokens = response.usage.output_tokens if response.usage else 0
        # input_tokens is the uncached remainder (total = input + cache_read + cache_creation).
        cached_tokens = 0
        cache_creation_tokens = 0
        if response.usage:
            cached_tokens = getattr(response.usage, 'cache_read_input_tokens', None) or 0
            cache_creation_tokens = getattr(response.usage, 'cache_creation_input_tokens', None) or 0

        result: dict[str, Any] = {
            'response': response,
            'content': content,
            'message': sanitized_message,  # Use sanitized message, not full response
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'cached_tokens': cached_tokens,
            'cache_creation_input_tokens': cache_creation_tokens,
            'fallback_used': served_model if is_fallback else None,
        }

        return result

    def _needs_direct_anthropic_call(self) -> bool:
        """Check if we need to bypass LiteLLM for direct Anthropic SDK call"""
        if 'anthropic' not in self.config.model_name:
            return False
        thinking = self.config.model_kwargs.get('thinking', {})
        # 'auto' and 'adaptive' thinking types require the direct SDK path because
        # LiteLLM does not preserve the `signature` field on thinking blocks when
        # they are passed back in conversation history, causing Anthropic to reject
        # them as "modified".
        return thinking.get('type') in ('auto', 'adaptive')

    @retry(
        stop=_length_aware_stop,
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
        if reasoning_param_name in actual_kwargs:
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
        if actual_kwargs.get('stream', False) and 'stream_options' not in actual_kwargs:
            actual_kwargs['stream_options'] = {'include_usage': True}

        messages_for_api = messages
        if 'anthropic' in self.config.model_name and messages:
            messages_for_api = copy.deepcopy(messages)
            last_message = messages_for_api[-1]
            _set_cache_breakpoint(last_message)
            for message in messages_for_api:
                if message.get('role') == 'system':
                    if message is not last_message:
                        _set_cache_breakpoint(message)
                    break

        try:
            res = litellm.completion(
                model=self.config.model_name, messages=messages_for_api, **actual_kwargs
            )

            if actual_kwargs.get('stream', False):
                chunks = []
                for chunk in res:
                    chunks.append(chunk)
                res = litellm.stream_chunk_builder(chunks, messages=messages_for_api)
        except litellm.exceptions.InternalServerError as e:
            if "This model's maximum context length is" in str(e):
                raise litellm.exceptions.ContextWindowExceededError(str(e), model=self.config.model_name, llm_provider=self.config.model_name) from e
            raise e

        if res['choices'][0]['finish_reason'] == 'length' and 'max_tokens' in actual_kwargs and res.usage.completion_tokens < actual_kwargs['max_tokens']:
            raise LengthFinishReasonError("Model returned length error but max tokens were not reached")

        content = res['choices'][0]['message']['content']

        # Native tool-callers (e.g. thinkingmachines/Inkling) put the action in
        # tool_calls and leave content empty. The agent loop is text-based, so
        # synthesize the equivalent ```bash block from bash tool calls.
        if not content:
            tool_calls = getattr(res['choices'][0]['message'], 'tool_calls', None) or []
            blocks = []
            for tc in tool_calls:
                fn = getattr(tc, 'function', None)
                if fn is not None and getattr(fn, 'name', '') == 'bash':
                    try:
                        cmd = json.loads(fn.arguments).get('command')
                    except (json.JSONDecodeError, TypeError, AttributeError):
                        cmd = None
                    if cmd:
                        blocks.append(f"```bash\n{cmd}\n```")
            if blocks:
                content = "\n".join(blocks)

        result = {
            'response': res,
            'message': res['choices'][0]['message']
        }
        if res and res.choices and len(res.choices) > 0:
            result['content'] = content
            result['input_tokens'] = res.usage.prompt_tokens
            result['output_tokens'] = res.usage.completion_tokens
            # OpenAI/Anthropic report cache reads under prompt_tokens_details; Gemini-via-litellm
            # uses a top-level field, read only as a fallback to avoid double-counting.
            result['cached_tokens'] = (
                _cached_tokens(getattr(res.usage, 'prompt_tokens_details', None))
                or getattr(res.usage, 'cache_read_input_tokens', 0) or 0
            )
            # Cache writes (Anthropic only)
            result['cache_creation_input_tokens'] = getattr(res.usage, 'cache_creation_input_tokens', 0) or 0
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
    def _responses_provider(self) -> str:
        try:
            return litellm.get_llm_provider(self.config.model_name)[1] or ''
        except Exception:
            return ''

    def _responses_supports_instructions(self) -> bool:
        """Whether the provider's Responses API accepts `instructions` (xAI rejects it)."""
        try:
            from litellm.utils import ProviderConfigManager
            model = self.config.model_name.split('/', 1)[-1]
            provider_config = ProviderConfigManager.get_provider_responses_api_config(self._responses_provider(), model)
            return provider_config is None or 'instructions' in provider_config.get_supported_openai_params(model)
        except Exception:
            return True

    def _query_responses(self, messages: list[dict[str, str]], replay_messages: list[dict[str, str]] | None = None, **kwargs):
        system_messages: list[str] = []
        for message in messages:
            if message.get('role') == 'system':
                system_messages.append(message.get('content', ''))
        system_prompt = system_messages[0] if system_messages else None

        # Stateful chaining sends only previous_response_id + the new turn; not universal:
        # OpenRouter's Responses API is stateless and ZDR OpenAI orgs reject
        # previous_response_id, so those use full-history replay instead.
        if self._responses_chaining is None and self._responses_provider() == 'openrouter':
            self._responses_chaining = False

        def _build_call(use_chaining: bool):
            call_kwargs = dict(self.config.model_kwargs | kwargs)
            use_instructions = system_prompt is not None and self._responses_supports_instructions()
            if use_chaining:
                call_kwargs['store'] = True
                if self.previous_response_id is not None:
                    src = messages[self._responses_sent_upto:]
                    call_kwargs['previous_response_id'] = self.previous_response_id
                else:
                    src = messages
                # without `instructions` support the system prompt must ride inside input
                roles = ('user', 'tool') if use_instructions else ('user', 'tool', 'system')
                input_to_send = [{'role': m['role'], 'content': m.get('content', '')}
                                 for m in src if m.get('role') in roles]
            else:
                input_to_send = replay_messages if replay_messages is not None else messages
                if use_instructions:
                    # system rides in `instructions`; keeping it in input double-counts it
                    input_to_send = [m for m in input_to_send
                                     if not (isinstance(m, dict) and m.get('role') == 'system')]
                if self._responses_provider() == 'openai':
                    # ZDR-canonical stateless replay: encrypted reasoning keeps replayed
                    # reasoning items valid instead of silently ignored
                    call_kwargs['store'] = False
                    existing_include = call_kwargs.get('include') or []
                    if 'reasoning.encrypted_content' not in existing_include:
                        call_kwargs['include'] = list(existing_include) + ['reasoning.encrypted_content']
            if use_instructions:
                call_kwargs['instructions'] = system_prompt
            return input_to_send, call_kwargs

        use_chaining = self._responses_chaining is not False
        input_to_send, call_kwargs = _build_call(use_chaining)
        try:
            res = litellm.responses(
                model=self.config.model_name, input=input_to_send, **call_kwargs,
            )
        except (litellm.exceptions.BadRequestError, litellm.exceptions.NotFoundError) as e:
            # A chaining call can fail because the provider rejects previous_response_id/store
            # (ZDR orgs), or because the referenced response is gone (xAI intermittently drops
            # stored responses -> NotFoundError). Either way, disable chaining for this instance
            # and replay the full history instead of erroring the run.
            err = str(e).lower()
            chaining_signal = any(s in err for s in (
                'previous_response_id', 'zero data retention', "'store'", 'not found', 'not-found'))
            if use_chaining and chaining_signal:
                logger.warning(
                    f"{self.config.model_name}: chaining rejected ({str(e)[:120]}); "
                    "disabling chaining, falling back to full replay"
                )
                self._responses_chaining = False
                self.previous_response_id = None
                input_to_send, call_kwargs = _build_call(False)
                res = litellm.responses(
                    model=self.config.model_name, input=input_to_send, **call_kwargs,
                )
            else:
                raise
        else:
            if use_chaining:
                self._responses_chaining = True
        self._responses_sent_upto = len(messages)
        self.previous_response_id = getattr(res, 'id', None)

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
            usage = res.usage
            result['input_tokens'] = getattr(usage, 'input_tokens', 0)
            result['output_tokens'] = getattr(usage, 'output_tokens', 0)
            # Responses API reports cache reads under input_tokens_details (not prompt_tokens_details).
            result['cached_tokens'] = (
                _cached_tokens(getattr(usage, 'input_tokens_details', None))
                or _cached_tokens(getattr(usage, 'prompt_tokens_details', None))
            )
        return result

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:

        actual_messages: list[dict[str, str]] = []
        for message in messages:
            message_copy = message.copy()
            if 'extra' in message_copy:
                if message_copy['extra'].get('message') is not None:
                    provider_fields = message_copy['extra']['message'].get('provider_specific_fields')
                    if provider_fields:
                        # flatten for reasoning_content/thinking_blocks consumers, but keep
                        # the nested dict too: litellm's gemini path reads thought_signatures
                        # from provider_specific_fields only
                        message_copy['extra']['message'] = message_copy['extra']['message'] | provider_fields
                    actual_messages.append(message_copy['extra']['message'])
                elif message_copy['extra'].get('outputs') is not None:
                    actual_messages.extend(message_copy['extra']['outputs'])
                else:
                    del message_copy['extra']
                    actual_messages.append(message_copy)
            else:
                actual_messages.append(message_copy)

        if self.config.native_gemini and 'gemini-3' in self.config.model_name:
            result = self._query_completion_generativeai(actual_messages, **kwargs)
        elif self._needs_direct_anthropic_call():
            result = self._query_completion_anthropic_direct(actual_messages, **kwargs)
        elif self.config.api_type == "completion":
            result = self._query_completion(actual_messages, **kwargs)
        elif self.config.api_type == "responses":
            # raw messages for chaining deltas; unpacked actual_messages for replay fallback
            result = self._query_responses(messages, replay_messages=actual_messages, **kwargs)
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
        self.cached_tokens += result.get('cached_tokens', 0) or 0
        self.cache_creation_tokens += result.get('cache_creation_input_tokens', 0) or 0

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
                # warnings=False: the server-side-fallback `usage.iterations` types aren't
                # fully modeled in anthropic SDK 0.105.2 and otherwise emit a noisy
                # PydanticSerializationUnexpectedValue warning on every fallback turn.
                "response": response.model_dump(warnings=False) if hasattr(response, 'model_dump') else response,
                "message": message_data,
                "outputs": result['outputs'] if 'outputs' in result else None,
                # record which turns were rescued by the server-side refusal fallback
                "fallback_used": result.get('fallback_used'),
            },
        }

        return res

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config) | {"n_model_calls": self.n_calls, "model_cost": self.cost, "total_input_tokens": self.input_tokens, "total_output_tokens": self.output_tokens}
