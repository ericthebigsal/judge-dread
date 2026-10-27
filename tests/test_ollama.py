import io
import json
from unittest.mock import patch
from urllib.error import URLError

import pytest
from google.genai import types

from judgedread.models import JudgeVerdict
from judgedread.ollama import OllamaClient, OllamaError


def _fake_response(payload: dict):
    return io.BytesIO(json.dumps(payload).encode())


def _generate_payload(text: str) -> dict:
    return {"response": text, "prompt_eval_count": 12, "eval_count": 34}


def test_generate_content_returns_text_and_token_counts():
    client = OllamaClient()

    with patch("judgedread.ollama.urlopen", return_value=_fake_response(_generate_payload("hi"))):
        response = client.models.generate_content(
            model="gemma3:4b",
            contents="Say hi",
            config=types.GenerateContentConfig(max_output_tokens=2048),
        )

    assert response.text == "hi"
    assert response.usage_metadata.prompt_token_count == 12
    assert response.usage_metadata.candidates_token_count == 34


def test_generate_content_sends_model_prompt_and_token_limit():
    client = OllamaClient(host="http://example:1234")

    with patch(
        "judgedread.ollama.urlopen", return_value=_fake_response(_generate_payload("hi"))
    ) as urlopen:
        client.models.generate_content(
            model="gemma3:4b",
            contents="Say hi",
            config=types.GenerateContentConfig(max_output_tokens=2048),
        )

    request = urlopen.call_args.args[0]
    body = json.loads(request.data)
    assert request.full_url == "http://example:1234/api/generate"
    assert body["model"] == "gemma3:4b"
    assert body["prompt"] == "Say hi"
    assert body["stream"] is False
    assert body["options"]["num_predict"] == 2048
    assert "format" not in body


def test_generate_content_with_response_schema_constrains_and_parses_output():
    client = OllamaClient()
    reply = _generate_payload('{"score": 4, "reasoning": "Good."}')

    with patch("judgedread.ollama.urlopen", return_value=_fake_response(reply)) as urlopen:
        response = client.models.generate_content(
            model="gemma3:4b",
            contents="Judge this",
            config=types.GenerateContentConfig(
                response_mime_type="application/json", response_schema=JudgeVerdict
            ),
        )

    body = json.loads(urlopen.call_args.args[0].data)
    assert body["format"] == JudgeVerdict.model_json_schema()
    assert response.parsed == JudgeVerdict(score=4, reasoning="Good.")


def test_generate_content_parsed_is_none_when_output_does_not_match_schema():
    client = OllamaClient()
    reply = _generate_payload("not json at all")

    with patch("judgedread.ollama.urlopen", return_value=_fake_response(reply)):
        response = client.models.generate_content(
            model="gemma3:4b",
            contents="Judge this",
            config=types.GenerateContentConfig(response_schema=JudgeVerdict),
        )

    assert response.parsed is None


def test_check_model_accepts_a_pulled_model():
    client = OllamaClient()
    tags = {"models": [{"name": "gemma3:4b"}, {"name": "mistral:latest"}]}

    with patch("judgedread.ollama.urlopen", side_effect=lambda *_: _fake_response(tags)):
        client.check_model("gemma3:4b")
        client.check_model("mistral")


def test_check_model_raises_when_model_is_not_pulled():
    client = OllamaClient()

    with (
        patch("judgedread.ollama.urlopen", return_value=_fake_response({"models": []})),
        pytest.raises(OllamaError, match="ollama pull gemma3:4b"),
    ):
        client.check_model("gemma3:4b")


def test_check_model_raises_when_ollama_is_not_running():
    client = OllamaClient()

    with (
        patch("judgedread.ollama.urlopen", side_effect=URLError("connection refused")),
        pytest.raises(OllamaError, match="Ollama is not reachable"),
    ):
        client.check_model("gemma3:4b")
