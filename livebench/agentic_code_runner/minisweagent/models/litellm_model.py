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


class GeminiContentFilterError(RuntimeError):
    """Gemini content filter aborted generation (finish_reason OTHER/RECITATION).
    Deterministic per prompt: not retried; fails the instance fast."""


class LengthFinishReasonError(Exception):
    """Raised when the model returns finish_reason='length' but completion_tokens < max_tokens."""


_DEFAULT_MAX_ATTEMPTS = 15
LENGTH_FINISH_REASON_MAX_ATTEMPTS = 3
# Per-request ceiling for the chat-completions path (maps to the httpx read timeout).
# Without it a provider that accepts the connection but never streams data back
# (observed with xAI/grok on large-context turns, 2026-07-28) hangs the read forever:
# the agent instance blocks, and because answers buffer in memory until the round
# finishes, ONE stuck question stalls the whole run. Bounding the request makes it
# fail fast -> retry -> error out, so the rest of the run survives. 600s matches the
# gemini genai path's http timeout; a real completion never legitimately needs longer.
_REQUEST_TIMEOUT_S = 600
# A provider that keeps timing out / refusing the connection will not recover on the
# 15th try, and each timed-out attempt already burned up to _REQUEST_TIMEOUT_S; cap these
# low so one bad endpoint can't spend _DEFAULT_MAX_ATTEMPTS x 600s (~2.5h) on a single
# question and stall the run. Built defensively: only exception classes present in this
# litellm version are matched (getattr → None otherwise), so it can never AttributeError.
_TIMEOUT_MAX_ATTEMPTS = 4
_TIMEOUT_EXC = tuple(c for c in (
    getattr(litellm.exceptions, "Timeout", None),
    getattr(litellm.exceptions, "APITimeoutError", None),
    getattr(litellm.exceptions, "APIConnectionError", None),
) if isinstance(c, type))


def _length_aware_stop(retry_state) -> bool:
    """tenacity stop: cap LengthFinishReasonError + timeout/connection retries early,
    retry everything else normally."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if isinstance(exc, LengthFinishReasonError):
        return retry_state.attempt_number >= LENGTH_FINISH_REASON_MAX_ATTEMPTS
    if _TIMEOUT_EXC and isinstance(exc, _TIMEOUT_EXC):
        return retry_state.attempt_number >= _TIMEOUT_MAX_ATTEMPTS
    return retry_state.attempt_number >= _DEFAULT_MAX_ATTEMPTS


def _cached_tokens(details) -> int:
    """Cached-token count from a token-details object (0 if absent)."""
    if details is None:
        return 0
    return getattr(details, 'cached_tokens', 0) or 0


# Native tool calling, always on for Anthropic models: the bash action is a real
# tool, so generation stops at the tool_use block and a turn can never run past its
# action into a hallucinated observation. The agent loop stays text-based: the
# tool_use command is synthesized back into a ```bash block for parse_action, and
# observation user-turns are wrapped into tool_result blocks at request-prep.
# Requires the tool-calling prompts in config/livebench_native.yaml (with
# tool_choice auto, triple-backtick prompts make models ignore the tool).
_BASH_TOOL = {
    "name": "bash",
    "description": (
        "Run a bash command in the repository environment and see its output. "
        "Each command runs in a fresh subshell: directory and environment-variable "
        "changes do not persist between calls (prefix with `cd /path && ...` as needed). "
        "Always use non-interactive flags; interactive tools are unavailable."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The bash command to execute."},
        },
        "required": ["command"],
    },
}

# Same bash tool in the OpenAI Responses API shape (flat function tool: type/name/
# description/parameters at top level, not nested under a `function` key like chat
# completions). Mirrors how the production client builds Responses tools.
_BASH_TOOL_RESPONSES = {
    "type": "function",
    "name": _BASH_TOOL["name"],
    "description": _BASH_TOOL["description"],
    "parameters": _BASH_TOOL["input_schema"],
}

# Chat-completions shape (nested under `function`) — what xAI/Grok (OpenAI-chat
# compatible, emits finish_reason=tool_calls) expects. Mirrors the production
# _to_xai_function_call_tool converter.
_BASH_TOOL_CHAT = {
    "type": "function",
    "function": {
        "name": _BASH_TOOL["name"],
        "description": _BASH_TOOL["description"],
        "parameters": _BASH_TOOL["input_schema"],
    },
}


def _wrap_tool_results(messages: list[dict]) -> None:
    """Wrap each user turn that follows a tool_use assistant turn in a tool_result
    block (the API requires every tool_use to be answered by one). Mutates in place;
    callers pass a request-only copy, never the agent's own history."""
    pending_tool_use_id = None
    for i, msg in enumerate(messages):
        if msg.get('role') == 'assistant':
            pending_tool_use_id = None
            blocks = msg.get('content')
            if isinstance(blocks, list):
                for block in blocks:
                    if isinstance(block, dict) and block.get('type') == 'tool_use':
                        pending_tool_use_id = block.get('id')
                        break
        elif msg.get('role') == 'user' and pending_tool_use_id is not None:
            content = msg.get('content')
            if isinstance(content, str):
                tool_result: dict[str, Any] = {'type': 'tool_result', 'tool_use_id': pending_tool_use_id}
                if content.strip():
                    tool_result['content'] = content
                messages[i] = {'role': 'user', 'content': [tool_result]}
            pending_tool_use_id = None


# Informative result for a tool_call we did not execute. The chat API requires a
# role:'tool' reply for EVERY tool_call_id in the preceding assistant turn; we run one
# command per turn, so extra calls get this. Mirrors mini-swe-agent, which pads skipped
# calls with returncode -1 + a note rather than a blank — the message also steers the
# model to stop emitting parallel calls (some providers ignore parallel_tool_calls=False).
_SKIPPED_TOOL_RESULT = (
    "<returncode>-1</returncode>\n<output>\n"
    "[skipped] Parallel tool calls are disabled: only the first command in a turn is "
    "executed. Issue exactly one command per turn.\n</output>"
)


def _tool_call_ids(msg: dict) -> list:
    ids = []
    for tc in (msg.get('tool_calls') or []):
        tc_id = tc.get('id') if isinstance(tc, dict) else getattr(tc, 'id', None)
        if tc_id:
            ids.append(tc_id)
    return ids


def _wrap_tool_results_chat(messages: list[dict]) -> None:
    """Chat-completions analog of _wrap_tool_results: an assistant `tool_calls` turn
    must be followed by a `role:'tool'` message for EACH tool_call id (OpenAI chat 400s
    otherwise: "tool_call_ids did not have response messages"). We execute one command
    per turn, so the observation answers the FIRST tool_call; any extra tool_calls the
    model emitted (providers such as Moonshot ignore parallel_tool_calls=False) are
    padded with an informative 'skipped' tool result keyed to their id — same shape as
    mini-swe-agent's pad-to-all-ids. Mutates a request-only copy in place."""
    result = []
    pending_ids: list = []
    for msg in messages:
        role = msg.get('role')
        if role == 'assistant':
            result.append(msg)
            pending_ids = _tool_call_ids(msg)
        elif role == 'user' and pending_ids:
            content = msg.get('content')
            if isinstance(content, str):
                result.append({'role': 'tool', 'tool_call_id': pending_ids[0],
                               'content': content if content.strip() else '(no output)'})
                for extra_id in pending_ids[1:]:
                    result.append({'role': 'tool', 'tool_call_id': extra_id,
                                   'content': _SKIPPED_TOOL_RESULT})
            else:
                result.append(msg)
            pending_ids = []
        else:
            result.append(msg)
    messages[:] = result


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
    # gemini-3 via the Interactions API (explicit content_blocked / incomplete
    # statuses instead of finish_reason guessing); False falls back to generateContent
    gemini_interactions: bool = True

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
        # turns whose action arrived as a tool_use block (vs regex-parsed text)
        self.native_tool_use_turns = 0
        if self.config.litellm_model_registry and Path(self.config.litellm_model_registry).is_file():
            litellm.utils.register_model(json.loads(Path(self.config.litellm_model_registry).read_text()))

    @retry(
        stop=stop_after_attempt(15),
        wait=wait_exponential(multiplier=2, min=4, max=120),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(
            (
                KeyboardInterrupt,
                GeminiContentFilterError,
            )
        ),
        reraise=True,
    )
    def _query_completion_interactions(self, messages: list[dict[str, str]], **kwargs):
        """Stateless step_list replay: prior model turns arrive as raw step dicts
        (via extra.outputs), so thought signatures are echoed verbatim."""
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                              http_options=types.HttpOptions(timeout=600_000))

        native = self._native_tools_enabled()
        system_parts, steps = [], []
        pending_call_id = None  # id of a replayed function_call awaiting its result (native)
        for m in messages:
            if not isinstance(m, dict) or 'type' in m:
                steps.append(m)  # replayed interaction step (incl function_call w/ signature)
                if native and isinstance(m, dict) and m.get('type') == 'function_call':
                    pending_call_id = m.get('id')
            elif m.get('role') == 'system':
                system_parts.append(m['content'])
            elif m.get('role') in ('assistant', 'model'):
                steps.append({"type": "model_output", "content": [{"type": "text", "text": m['content']}]})
            elif native and pending_call_id is not None:
                # observation answering a function_call -> function_result step (call_id-paired)
                steps.append({"type": "function_result", "call_id": pending_call_id,
                              "name": _BASH_TOOL['name'], "result": m['content']})
                pending_call_id = None
            else:
                steps.append({"type": "user_input", "content": [{"type": "text", "text": m['content']}]})

        gc = dict(self.config.model_kwargs) | kwargs
        tc = gc.pop('thinking_config', None) or {}
        if tc.get('thinking_level'):
            gc['thinking_level'] = tc['thinking_level']
        for k in ('stream', 'timeout', 'safety_settings'):
            gc.pop(k, None)

        create_kwargs: dict[str, Any] = dict(
            model=self.config.model_name.split('/')[-1],
            input=steps,
            system_instruction="\n".join(system_parts) or None,
            generation_config=gc,
        )
        if native:
            # bash as an interactions function tool (flat {type,name,description,parameters});
            # tool_config isn't accepted here, so tool_choice is effectively auto.
            create_kwargs['tools'] = [{
                "type": "function", "name": _BASH_TOOL['name'], "description": _BASH_TOOL['description'],
                "parameters": {"type": "OBJECT",
                               "properties": {"command": {"type": "STRING", "description": "The bash command to execute."}},
                               "required": ["command"]},
            }]

        try:
            response = client.interactions.create(**create_kwargs)
        except Exception as e:
            s = str(e)
            if 'content_blocked' in s or 'blocked for' in s:
                logger.error(f"Gemini content filter block: {s[:200]}")
                raise GeminiContentFilterError(f"Gemini content policy block: {s[:200]}")
            raise

        out_steps = response.model_dump().get('steps') or []
        text = "".join(c.get('text', '') for s in out_steps if s.get('type') == 'model_output'
                       for c in (s.get('content') or []) if c.get('type') == 'text')

        tool_command = None
        if native:
            for s in out_steps:
                if s.get('type') == 'function_call' and s.get('name') == _BASH_TOOL['name']:
                    args = s.get('arguments') or {}
                    if isinstance(args, str):
                        try: args = json.loads(args)
                        except Exception: args = {}
                    tool_command = args.get('command') if isinstance(args, dict) else None
                    if tool_command:
                        break

        if native and tool_command:
            pass  # valid function-call turn (status is 'requires_action')
        elif response.status == 'incomplete':  # output budget exhausted (usually by thinking)
            if not text:
                self.empty_responses += 1
                logger.warning("Gemini interaction incomplete with no text; returning empty content")
        elif response.status != 'completed' or not text:
            raise Exception(f"Empty interaction response (status={response.status})")

        if native and tool_command and tool_command.strip():
            # synthesize the ```bash block for the text agent loop; raw command carried
            # out-of-band via result['tool_command'] for verbatim execution
            self.native_tool_use_turns += 1
            action_block = f"```bash\n{tool_command}\n```"
            text = (text.replace("```bash", "```sh") + f"\n\n{action_block}") if text.strip() else action_block

        usage = response.usage
        return {
            'response': response,
            'content': text,
            'outputs': out_steps,  # replayed verbatim next turn, signatures included
            'tool_command': tool_command if native else None,
            'input_tokens': getattr(usage, 'total_input_tokens', 0) or 0,
            'output_tokens': (getattr(usage, 'total_output_tokens', 0) or 0)
                             + (getattr(usage, 'total_thought_tokens', 0) or 0),
            'cached_tokens': getattr(usage, 'total_cached_tokens', 0) or 0,
        }

    @retry(
        stop=stop_after_attempt(15),
        wait=wait_exponential(multiplier=2, min=4, max=120),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(
            (
                KeyboardInterrupt,
                GeminiContentFilterError,
            )
        ),
        reraise=True,
    )
    def _query_completion_generativeai(self, messages: list[dict[str, str]], **kwargs):
        from google import genai
        from google.genai import types
        # 600s hard deadline: Gemini can hold a connection open indefinitely
        # on filter-triggering prompts, pinning the worker.
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                              http_options=types.HttpOptions(timeout=600_000))
        native = self._native_tools_enabled()
        actual_messages: list[types.ContentOrDict] = []
        system = None
        pending_fn = None  # name of a model function_call awaiting its response (native pairing)
        for message in messages:
            role = message['role']
            if role == 'system':
                if system is None:
                    system = types.Content(role='system', parts=[types.Part.from_text(text=message['content'])])
                else:
                    system.parts.append(types.Part.from_text(text=message['content']))
            elif role == 'user':
                if native and pending_fn is not None:
                    # observation answering a function_call -> function_response part
                    actual_messages.append(types.Content(role='user', parts=[
                        types.Part.from_function_response(name=pending_fn, response={'output': message['content']})]))
                    pending_fn = None
                else:
                    actual_messages.append(types.Content(role='user', parts=[types.Part.from_text(text=message['content'])]))
            elif role in ('assistant', 'model'):
                message['role'] = 'model'
                actual_messages.append(message)  # stored model Content (parts incl thought sig + function_call) replayed verbatim
                if native:
                    pending_fn = None
                    for p in (message.get('parts') or []):
                        fc = p.get('function_call') if isinstance(p, dict) else getattr(p, 'function_call', None)
                        if fc:
                            pending_fn = fc.get('name') if isinstance(fc, dict) else getattr(fc, 'name', None)
                            break

        safety_settings = [
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY, threshold=types.HarmBlockThreshold.BLOCK_NONE),
        ]

        api_kwargs = self.config.model_kwargs | kwargs

        config_kwargs: dict[str, Any] = dict(safety_settings=safety_settings, system_instruction=system, **api_kwargs)
        if native:
            # bash exposed as a genai function tool (cleaned schema per production
            # _to_gemini_function_call_tool: no additionalProperties/strict). tool_choice
            # is AUTO by default when tools are present.
            config_kwargs['tools'] = [{'function_declarations': [{
                'name': _BASH_TOOL['name'],
                'description': _BASH_TOOL['description'],
                'parameters': {
                    'type': 'OBJECT',
                    'properties': {'command': {'type': 'STRING', 'description': 'The bash command to execute.'}},
                    'required': ['command'],
                },
            }]}]
            # force a function call each turn (mode ANY) so Gemini genuinely goes native
            # instead of following the instance_template's ```bash-text convention.
            config_kwargs['tool_config'] = {'function_calling_config': {'mode': 'ANY'}}
        config = types.GenerateContentConfig(**config_kwargs)

        actual_model_name = self.config.model_name.split('/')[-1]

        # Empty text: resample briefly, then return "" so a step is counted
        # (raising into tenacity's 4..120s backoff hides it from the agent).
        max_empty_resamples = 3
        tool_command = None
        for _empty_attempt in range(max_empty_resamples + 1):
            response = client.models.generate_content(model=actual_model_name, contents=actual_messages, config=config)

            if response.candidates is None or len(response.candidates) == 0:
                raise Exception(
                    "No response returned from Google: no candidates "
                    f"(prompt_feedback={getattr(response, 'prompt_feedback', None)})"
                )

            if native:
                # a function_call is a valid response even when response.text is empty
                for p in (response.candidates[0].content.parts or []):
                    fc = getattr(p, 'function_call', None)
                    if fc and getattr(fc, 'name', '') == 'bash':
                        _a = dict(getattr(fc, 'args', {}) or {})
                        tool_command = _a.get('command')
                        if tool_command:
                            break
                if tool_command:
                    message = response.text or ""
                    if _empty_attempt > 0:
                        self.empty_resamples_recovered += 1
                    break

            if response.text is not None:
                message = response.text
                if _empty_attempt > 0:
                    self.empty_resamples_recovered += 1
                    logger.warning(f"Recovered from empty Gemini response after {_empty_attempt} resample(s)")
                break

            cand = response.candidates[0]
            finish = str(getattr(cand, 'finish_reason', ''))
            um = response.usage_metadata
            thoughts = getattr(um, 'thoughts_token_count', None) if um is not None else None
            self.empty_responses += 1

            if finish.endswith('OTHER') or finish.endswith('RECITATION'):
                # Server-side content filter; deterministic per prompt, so
                # exit early instead of retrying. ("content policy" in the
                # message makes validate_eval classify it as terminal.)
                logger.error(f"Gemini content-filter abort (finish_reason={finish}); failing instance")
                raise GeminiContentFilterError(
                    f"Gemini content policy block: finish_reason={finish}, generation aborted server-side"
                )

            if finish.endswith('MAX_TOKENS'):
                # Whole output budget went to thinking: token exhaustion, don't retry.
                logger.warning(f"Gemini thought-only response (thoughts_tokens={thoughts}); returning empty content")
                message = ""
                break

            if _empty_attempt < max_empty_resamples:
                backoff = min(2 ** _empty_attempt, 8)
                logger.warning(
                    f"Empty text response from Google (finish_reason={finish}, "
                    f"thoughts_tokens={thoughts}); resampling "
                    f"(attempt {_empty_attempt + 1}/{max_empty_resamples}, sleeping {backoff}s)"
                )
                time.sleep(backoff)
                continue

            logger.warning(
                f"Empty text response from Google persisted after {max_empty_resamples} "
                f"resamples (finish_reason={finish}); returning empty content"
            )
            message = ""

        if native and tool_command and tool_command.strip():
            # synthesize the ```bash block for the text agent loop; raw command carried
            # out-of-band via result['tool_command'] for verbatim execution
            self.native_tool_use_turns += 1
            action_block = f"```bash\n{tool_command}\n```"
            if message and message.strip():
                message = message.replace("```bash", "```sh") + f"\n\n{action_block}"
            else:
                message = action_block

        result: dict[str, Any] = {
            'response': response,
            'content': message,
            'message': response.candidates[0].content,
            'tool_command': tool_command,
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

        from livebench.model.completions import anthropic_api_key

        client = Anthropic(api_key=anthropic_api_key(self.config.model_name))
        
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
        
        native_tools = self._native_tools_enabled()
        if native_tools:
            _wrap_tool_results(actual_messages)

        # Build call kwargs - only include supported parameters
        call_kwargs: dict[str, Any] = {
            'model': actual_model_name,
            'messages': actual_messages,
            'max_tokens': max_tokens,
        }
        if native_tools:
            call_kwargs['tools'] = [_BASH_TOOL]
            # at most one tool_use per turn: one-command-per-turn, API-enforced
            call_kwargs['tool_choice'] = {'type': 'auto', 'disable_parallel_tool_use': True}

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
            tool_command = None
            for block in response.content:
                block_type = getattr(block, 'type', 'unknown')

                if block_type == "tool_use":
                    if tool_command is not None:
                        # keeps the 1 tool_use : 1 tool_result pairing valid
                        logger.warning("Multiple tool_use blocks in one turn; keeping only the first")
                        continue
                    tool_input = getattr(block, 'input', {}) or {}
                    sanitized_content.append({
                        "type": "tool_use",
                        "id": getattr(block, 'id', ''),
                        "name": getattr(block, 'name', ''),
                        "input": tool_input,
                    })
                    tool_command = tool_input.get('command', '') if getattr(block, 'name', '') == 'bash' else ''
                elif block_type == "text":
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

            if tool_command is not None and tool_command.strip():
                # synthesize the ```bash block parse_action expects; the raw tool_use
                # block is what gets replayed to the API
                self.native_tool_use_turns += 1
                action_block = f"```bash\n{tool_command}\n```"
                if content:
                    # neutralize stray fences so parse_action picks the tool call
                    content = content.replace("```bash", "```sh")
                    content = f"{content}\n\n{action_block}"
                else:
                    content = action_block

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
            # Raw tool_use command carried out-of-band so parse_action executes it
            # verbatim instead of re-parsing the synthesized ```bash block (whose
            # non-greedy regex truncates any command body containing a code fence).
            'tool_command': tool_command,
            'message': sanitized_message,  # Use sanitized message, not full response
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'cached_tokens': cached_tokens,
            'cache_creation_input_tokens': cache_creation_tokens,
            'fallback_used': served_model if is_fallback else None,
        }

        return result

    def _native_tools_enabled(self) -> bool:
        """Native tool calling: bash exposed as a real tool instead of regex-parsed
        text. Always on for Anthropic models (direct SDK path). Also on for OpenAI
        Responses-API models (gpt-5.x-sol etc.), which support native function tools
        + preserved reasoning items -- toggleable via MSWEA_OPENAI_NATIVE_TOOLS."""
        # One uniform switch across all families (default on). Each family exposes the
        # SAME bash tool via its provider-native schema + tool_choice='auto', with a
        # graceful text-```bash fallback; native_tool_use_turns records actual adherence.
        if os.getenv("MSWEA_NATIVE_TOOLS", "1") != "1":
            return False
        mn = self.config.model_name
        if 'anthropic' in mn:                                                   # Anthropic direct SDK (tool_use)
            return True
        if self.config.api_type == "responses" and self._responses_provider() == "openai":  # OpenAI Responses (function_call)
            return True
        if "grok" in mn and self.config.api_type == "completion":               # Grok/xAI chat (tool_calls); rewritten to openai/grok-*
            return True
        if ("glm" in mn or "kimi" in mn) and self.config.api_type == "completion":  # GLM (z.ai) / Kimi (moonshot): OpenAI-compatible chat, share the grok tool_calls path (tool_choice=required)
            return True
        if "qwen" in mn and self.config.api_type == "completion":               # Alibaba DashScope OpenAI-compat chat (tool_calls); tool_choice must be 'auto' (thinking mode)
            return True
        if "inkling" in mn.lower() and self.config.api_type == "completion":     # Thinking Machines Inkling (Tinker OpenAI-compat chat, self-invoking tool_calls); same tool_calls path
            return True
        if self.config.native_gemini and 'gemini-3' in mn:                      # Gemini genai generate_content (function_call parts)
            return True
        return False

    def _needs_direct_anthropic_call(self) -> bool:
        """Check if we need to bypass LiteLLM for direct Anthropic SDK call"""
        if 'anthropic' not in self.config.model_name:
            return False
        if self._native_tools_enabled():
            # tool_use/tool_result plumbing exists only on the direct path
            return True
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
        # provider_specific_fields is a litellm replay artifact (the nested reasoning /
        # thought-signature payload). It is NOT a valid OpenAI chat-completion input field:
        # query() already flattens reasoning_content/thinking_blocks to the top level, and
        # the nested key is only needed by the Gemini path (separate) and Anthropic-via-
        # litellm (kept below). Strict OpenAI-compatible providers (e.g. Fireworks) 400 with
        # "Extra inputs are not permitted, field: messages[N].provider_specific_fields", which
        # kills the agent after turn 1. The messages here are fresh per-query dicts (see
        # query()), so popping in place does not corrupt the stored history.
        if 'anthropic' not in self.config.model_name:
            for message in messages:
                message.pop('provider_specific_fields', None)
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
        # Bound every request so a hung provider stream can't block the run forever
        # (see _REQUEST_TIMEOUT_S). setdefault so an explicit config timeout wins.
        actual_kwargs.setdefault('timeout', _REQUEST_TIMEOUT_S)

        native = self._native_tools_enabled()
        if native:
            actual_kwargs['tools'] = [_BASH_TOOL_CHAT]
            # force a tool call each turn (grok otherwise follows the ```bash-text prompt);
            # 'auto' where the provider rejects a forced choice in thinking mode (see helper)
            actual_kwargs['tool_choice'] = self._native_tool_choice()
            # one command per turn: _wrap_tool_results_chat answers only the first
            # tool_call, so a parallel emission would leave an unanswered tool_call_id
            # and 400 on the next request. Disable parallel calls where the provider
            # honors it (OpenAI-compatible: OpenAI/xAI/z.ai/moonshot).
            actual_kwargs['parallel_tool_calls'] = False

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
        if native:
            # each observation must be a role:'tool' message keyed on its tool_call_id
            if messages_for_api is messages:
                messages_for_api = copy.deepcopy(messages)
            _wrap_tool_results_chat(messages_for_api)

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
        tool_command = None

        # Native tool call (grok/xai chat, or self-invoking models like Inkling): the
        # action arrives in tool_calls. Take the first bash command, synthesize the
        # ```bash block the text agent loop expects, and carry the raw command
        # out-of-band (result['tool_command']) for verbatim execution.
        tool_calls = getattr(res['choices'][0]['message'], 'tool_calls', None) or []
        for tc in tool_calls:
            fn = getattr(tc, 'function', None)
            if fn is not None and getattr(fn, 'name', '') == 'bash':
                try:
                    tool_command = json.loads(fn.arguments).get('command')
                except (json.JSONDecodeError, TypeError, AttributeError):
                    tool_command = None
                if tool_command:
                    break
        if tool_command and tool_command.strip():
            self.native_tool_use_turns += 1
            action_block = f"```bash\n{tool_command}\n```"
            if content and content.strip():
                content = content.replace("```bash", "```sh") + f"\n\n{action_block}"
            else:
                content = action_block

        result = {
            'response': res,
            'message': res['choices'][0]['message'],
            'tool_command': tool_command,
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

    def _native_tool_choice(self) -> str:
        """tool_choice for the native tool paths (both Responses and chat-completions):
        'required' where the provider allows forcing a call, else 'auto'.

        Some providers reject a forced choice while thinking is on — and on these models
        thinking CANNOT be disabled, so 'required' can never be used there:
          * DeepSeek's own endpoint  -> 400 "Thinking mode does not support this tool_choice"
            (api.deepseek.com lists no non-thinking slug)
          * Alibaba DashScope (qwen3.x-max) -> 400 "The tool_choice parameter does not support
            being set to required or object in thinking mode" (enable_thinking=False is
            ignored — reasoning_content comes back regardless)
        Gated on the ENDPOINT, not the model name: the same DeepSeek weights served by
        Fireworks DO accept 'required' (verified 200), and keeping the forced choice where it
        works is strictly better. Under 'auto' these models still emit a bash tool call nearly
        every turn, so the run stays genuinely native; the text-```bash fallback covers a miss
        and native_tool_use_turns records the real adherence.
        """
        api_base = str(self.config.model_kwargs.get('api_base') or '').lower()
        if any(h in api_base for h in ('deepseek', 'dashscope', 'aliyuncs')):
            return 'auto'
        return 'required'

    def _responses_supports_instructions(self) -> bool:
        """Whether the provider's Responses API accepts `instructions` (xAI rejects it)."""
        try:
            from litellm.utils import ProviderConfigManager
            model = self.config.model_name.split('/', 1)[-1]
            provider_config = ProviderConfigManager.get_provider_responses_api_config(self._responses_provider(), model)
            return provider_config is None or 'instructions' in provider_config.get_supported_openai_params(model)
        except Exception:
            return True

    @staticmethod
    def _content_text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for b in content:
                if isinstance(b, dict):
                    parts.append(b.get('text') or b.get('output_text') or '')
                elif isinstance(b, str):
                    parts.append(b)
            return ''.join(parts)
        return ''

    def _responses_native_input(self, msgs: list, include_system: bool) -> list:
        """Translate the agent's history into OpenAI Responses `input` items for
        native tool use. Prior assistant turns arrive as raw output items (reasoning /
        function_call / message, replayed via extra.outputs) and pass through verbatim
        so encrypted reasoning + call_ids stay valid. Each user observation that answers
        a function_call becomes a `function_call_output` keyed on that call_id (FIFO,
        one command per turn); the initial task and any non-tool user text stay user
        messages. Mirrors the production client's _message_to_items pairing."""
        items: list = []
        pending_call_id = None
        for m in msgs:
            if not isinstance(m, dict):
                continue
            itype = m.get('type')
            if itype in ('reasoning', 'function_call', 'function_call_output', 'message', 'web_search_call'):
                items.append(m)
                if itype == 'function_call':
                    pending_call_id = m.get('call_id')
                continue
            role = m.get('role')
            text = self._content_text(m.get('content'))
            if role == 'system':
                if include_system and text:
                    items.append({'type': 'message', 'role': 'system',
                                  'content': [{'type': 'input_text', 'text': text}]})
            elif role == 'user':
                if pending_call_id is not None:
                    items.append({'type': 'function_call_output', 'call_id': pending_call_id, 'output': text})
                    pending_call_id = None
                elif text:
                    items.append({'type': 'message', 'role': 'user',
                                  'content': [{'type': 'input_text', 'text': text}]})
            elif role == 'assistant':
                # non-native assistant text (e.g. an empty-degrade turn with no tool call)
                if text:
                    items.append({'type': 'message', 'role': 'assistant',
                                  'content': [{'type': 'output_text', 'text': text}]})
        return items

    def _query_responses(self, messages: list[dict[str, str]], replay_messages: list[dict[str, str]] | None = None, **kwargs):
        system_messages: list[str] = []
        for message in messages:
            if message.get('role') == 'system':
                system_messages.append(message.get('content', ''))
        system_prompt = system_messages[0] if system_messages else None
        native = self._native_tools_enabled()

        # Stateful chaining sends only previous_response_id + the new turn; not universal:
        # OpenRouter's Responses API is stateless and ZDR OpenAI orgs reject
        # previous_response_id, so those use full-history replay instead.
        if self._responses_chaining is None and self._responses_provider() == 'openrouter':
            self._responses_chaining = False
        # Native tool use rebuilds the full history each turn (matched function_call /
        # function_call_output items + encrypted reasoning); the chaining path's
        # server-side call pairing isn't modeled here, so force stateless replay.
        if native:
            self._responses_chaining = False

        def _build_call(use_chaining: bool):
            call_kwargs = dict(self.config.model_kwargs | kwargs)
            use_instructions = system_prompt is not None and self._responses_supports_instructions()
            if native:
                call_kwargs['tools'] = [_BASH_TOOL_RESPONSES]
                # force a tool call each turn so the run is genuinely native (the shared
                # instance_template asks for ```bash text, which models otherwise follow);
                # submit is itself a bash call, so forcing loses nothing.
                call_kwargs['tool_choice'] = self._native_tool_choice()
                call_kwargs['parallel_tool_calls'] = False
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
                base = replay_messages if replay_messages is not None else messages
                if native:
                    # translate history -> Responses items, pairing each observation to
                    # its function_call via call_id (FIFO), like the production client
                    input_to_send = self._responses_native_input(base, include_system=not use_instructions)
                else:
                    input_to_send = base
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
        tool_command = None

        for output_item in res.output:
            output_item = output_item.model_dump() if not isinstance(output_item, dict) else output_item
            outputs.append(output_item)
            itype = output_item.get('type')
            if itype == 'message':
                for content in output_item.get('content', []):
                    if content.get('type') == 'output_text':
                        output_text += content.get('text', '')
            elif itype == 'function_call' and native and tool_command is None:
                # one command per turn (parallel_tool_calls=False); take the first bash call
                if output_item.get('name') == 'bash':
                    try:
                        args = json.loads(output_item.get('arguments') or '{}')
                    except Exception:
                        args = {}
                    tool_command = args.get('command', '')

        content_out = output_text
        if native and tool_command is not None and tool_command.strip():
            # synthesize the ```bash block parse_action expects; the raw command is
            # also carried out-of-band via result['tool_command'] (verbatim execution)
            self.native_tool_use_turns += 1
            action_block = f"```bash\n{tool_command}\n```"
            if output_text.strip():
                content_out = output_text.replace("```bash", "```sh") + f"\n\n{action_block}"
            else:
                content_out = action_block

        result = {
            'response': res,
            'content': content_out,
            'outputs': outputs,
        }
        if native:
            result['tool_command'] = tool_command
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
                    actual_messages.append(dict(message_copy['extra']['message']))
                elif message_copy['extra'].get('outputs') is not None:
                    actual_messages.extend(message_copy['extra']['outputs'])
                else:
                    del message_copy['extra']
                    actual_messages.append(message_copy)
            else:
                actual_messages.append(message_copy)

        if self.config.native_gemini and 'gemini-3' in self.config.model_name:
            # Native tools -> generate_content: it's the only Gemini path that can FORCE
            # tool use (tool_config mode=ANY); the Interactions API supports tools too and
            # has nicer status/signature handling, but rejects tool_config, so under the
            # ```bash-text instance_template Gemini won't actually call the tool there
            # (verified: native_turns=0). Non-native runs keep the Interactions default.
            if self.config.gemini_interactions and not self._native_tools_enabled():
                result = self._query_completion_interactions(actual_messages, **kwargs)
            else:
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
            # Raw native tool command carried out-of-band so the agent's parse_action
            # executes it verbatim (avoids the ```bash regex round-trip truncating a
            # command body that itself contains a code fence). None on non-native turns.
            "tool_command": result.get("tool_command"),
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
