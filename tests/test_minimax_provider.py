"""Tests for MiniMax provider configs (minimax-m2.7 and minimax-m2.7-highspeed)."""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from livebench.model.api_model_config import load_model_configs, load_all_configs

MINIMAX_YML = os.path.join(
    os.path.dirname(__file__),
    '../livebench/model/model_configs/minimax.yml'
)


def test_minimax_m2_7_config_loads():
    """minimax-m2.7 config should load from minimax.yml."""
    configs = load_model_configs(MINIMAX_YML)
    assert 'minimax-m2.7' in configs


def test_minimax_m2_7_highspeed_config_loads():
    """minimax-m2.7-highspeed config should load from minimax.yml."""
    configs = load_model_configs(MINIMAX_YML)
    assert 'minimax-m2.7-highspeed' in configs


def test_minimax_m2_7_api_name():
    """minimax-m2.7 should map to MiniMax-M2.7 on the MiniMax API."""
    configs = load_model_configs(MINIMAX_YML)
    cfg = configs['minimax-m2.7']
    assert 'https://api.minimax.io/v1' in cfg.api_name
    assert cfg.api_name['https://api.minimax.io/v1'] == 'MiniMax-M2.7'


def test_minimax_m2_7_highspeed_api_name():
    """minimax-m2.7-highspeed should map to MiniMax-M2.7-highspeed on the MiniMax API."""
    configs = load_model_configs(MINIMAX_YML)
    cfg = configs['minimax-m2.7-highspeed']
    assert 'https://api.minimax.io/v1' in cfg.api_name
    assert cfg.api_name['https://api.minimax.io/v1'] == 'MiniMax-M2.7-highspeed'


def test_minimax_m2_7_api_key_env_var():
    """minimax-m2.7 should use MINIMAX_API_KEY env var."""
    configs = load_model_configs(MINIMAX_YML)
    cfg = configs['minimax-m2.7']
    assert cfg.api_keys is not None
    assert cfg.api_keys.get('https://api.minimax.io/v1') == 'MINIMAX_API_KEY'


def test_minimax_m2_7_highspeed_api_key_env_var():
    """minimax-m2.7-highspeed should use MINIMAX_API_KEY env var."""
    configs = load_model_configs(MINIMAX_YML)
    cfg = configs['minimax-m2.7-highspeed']
    assert cfg.api_keys is not None
    assert cfg.api_keys.get('https://api.minimax.io/v1') == 'MINIMAX_API_KEY'


def test_minimax_m2_7_temperature_default():
    """minimax-m2.7 default temperature should be 1."""
    configs = load_model_configs(MINIMAX_YML)
    cfg = configs['minimax-m2.7']
    assert cfg.api_kwargs is not None
    assert cfg.api_kwargs['default']['temperature'] == 1


def test_minimax_m2_7_base_url_is_overseas():
    """minimax-m2.7 base URL must be the overseas api.minimax.io, not api.minimax.chat."""
    configs = load_model_configs(MINIMAX_YML)
    cfg = configs['minimax-m2.7']
    for key in cfg.api_name:
        assert 'api.minimax.io' in key, f"Expected api.minimax.io in URL, got: {key}"
        assert 'api.minimax.chat' not in key


def test_all_configs_include_minimax_m2_7():
    """Global model registry should include minimax-m2.7 after loading all configs."""
    configs = load_all_configs()
    assert 'minimax-m2.7' in configs
    assert 'minimax-m2.7-highspeed' in configs


@pytest.mark.skipif(
    not os.environ.get('MINIMAX_API_KEY'),
    reason='MINIMAX_API_KEY not set'
)
def test_minimax_m2_7_integration():
    """Integration test: call MiniMax-M2.7 via the OpenAI-compatible API."""
    from livebench.model.completions import chat_completion_openai

    api_dict = {
        'api_base': 'https://api.minimax.io/v1',
        'api_key': os.environ['MINIMAX_API_KEY'],
    }
    output, num_tokens, _ = chat_completion_openai(
        model='MiniMax-M2.7',
        messages=[{'role': 'user', 'content': 'Reply with the single word: hello'}],
        temperature=1.0,
        max_tokens=20,
        api_dict=api_dict,
    )
    assert output and output != '$ERROR$', f"Got error output: {output}"
    assert num_tokens >= 0


@pytest.mark.skipif(
    not os.environ.get('MINIMAX_API_KEY'),
    reason='MINIMAX_API_KEY not set'
)
def test_minimax_m2_7_highspeed_integration():
    """Integration test: call MiniMax-M2.7-highspeed via the OpenAI-compatible API."""
    from livebench.model.completions import chat_completion_openai

    api_dict = {
        'api_base': 'https://api.minimax.io/v1',
        'api_key': os.environ['MINIMAX_API_KEY'],
    }
    output, num_tokens, _ = chat_completion_openai(
        model='MiniMax-M2.7-highspeed',
        messages=[{'role': 'user', 'content': 'Reply with the single word: hello'}],
        temperature=1.0,
        max_tokens=20,
        api_dict=api_dict,
    )
    assert output and output != '$ERROR$', f"Got error output: {output}"
    assert num_tokens >= 0
