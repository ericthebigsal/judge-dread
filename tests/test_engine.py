import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from judgedread.engine import (
    hydrate_prompt,
    invoke_model,
    load_prompt_template,
    load_test_suite,
    run_test_case,
)
from judgedread.models import TestCase


def _fake_response(text: str, input_tokens: int = 12, output_tokens: int = 34):
    return SimpleNamespace(
        text=text,
        usage_metadata=SimpleNamespace(
            prompt_token_count=input_tokens, candidates_token_count=output_tokens
        ),
    )


def test_load_test_suite_parses_json_file(tmp_path):
    suite_path = tmp_path / "test_suite.json"
    suite_path.write_text(
        json.dumps(
            {
                "suite_name": "Suite",
                "target_prompt_path": "prompts/system_prompt.txt",
                "test_cases": [
                    {"id": "TC_001", "description": "d", "input_variables": {}}
                ],
            }
        )
    )

    suite = load_test_suite(str(suite_path))

    assert suite.suite_name == "Suite"
    assert suite.test_cases[0].id == "TC_001"


def test_load_prompt_template_reads_file_text(tmp_path):
    prompt_path = tmp_path / "system_prompt.txt"
    prompt_path.write_text("Respond to: ${user_query}")

    text = load_prompt_template(str(prompt_path))

    assert text == "Respond to: ${user_query}"


def test_hydrate_prompt_substitutes_variables():
    template = "Build an endpoint for: ${user_query}. Format: ${fmt}"
    result = hydrate_prompt(template, {"user_query": "profile creation", "fmt": "json"})
    assert result == "Build an endpoint for: profile creation. Format: json"


def test_hydrate_prompt_raises_on_missing_variable():
    template = "Needs ${missing_var}"
    with pytest.raises(KeyError):
        hydrate_prompt(template, {})


def test_invoke_model_returns_text_latency_and_token_counts():
    client = Mock()
    client.models.generate_content.return_value = _fake_response(
        '{"path": "/profile"}', input_tokens=50, output_tokens=75
    )

    text, latency_ms, input_tokens, output_tokens = invoke_model(
        client, "some prompt", model="gemini-2.5-flash"
    )

    assert text == '{"path": "/profile"}'
    assert latency_ms >= 0
    assert input_tokens == 50
    assert output_tokens == 75
    client.models.generate_content.assert_called_once()
    _, kwargs = client.models.generate_content.call_args
    assert kwargs["model"] == "gemini-2.5-flash"
    assert kwargs["contents"] == "some prompt"


def test_run_test_case_hydrates_then_invokes():
    client = Mock()
    client.models.generate_content.return_value = _fake_response("output text")
    test_case = TestCase.model_validate(
        {
            "id": "TC_001",
            "description": "d",
            "input_variables": {"user_query": "build a thing"},
        }
    )
    template_text = "Task: ${user_query}"

    text, latency_ms, input_tokens, output_tokens = run_test_case(
        client, test_case, template_text
    )

    assert text == "output text"
    _, kwargs = client.models.generate_content.call_args
    assert kwargs["contents"] == "Task: build a thing"
