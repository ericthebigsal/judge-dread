import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

from judgedread.cli import build_results, main
from judgedread.models import JudgeVerdict, TestSuite
from judgedread.ollama import OllamaError


def _fake_generate_content_response(text: str):
    return SimpleNamespace(
        text=text,
        usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=20),
    )


def test_build_results_runs_deterministic_and_judge_checks():
    suite = TestSuite.model_validate(
        {
            "suite_name": "Suite",
            "target_prompt_path": "unused.txt",
            "test_cases": [
                {
                    "id": "TC_001",
                    "description": "Verify JSON output",
                    "input_variables": {"user_query": "build an endpoint"},
                    "assertions": {
                        "expected_format": "json",
                        "required_keys": ["path"],
                    },
                    "rubric": {
                        "metric": "technical_accuracy",
                        "criteria": "Is the path field present and correct?",
                        "passing_threshold": 4,
                    },
                }
            ],
        }
    )
    template_text = "Task: ${user_query}"

    client = Mock()
    client.models.generate_content.side_effect = [
        _fake_generate_content_response('{"path": "/profile"}'),
        SimpleNamespace(parsed=JudgeVerdict(score=5, reasoning="Path is correct.")),
    ]

    results = build_results(suite, template_text, client)

    assert len(results) == 1
    result = results[0]
    assert result.id == "TC_001"
    assert result.deterministic.passed is True
    assert result.judge_verdict.score == 5
    assert result.status == "PASS"


def test_build_results_fails_when_judge_score_below_threshold():
    suite = TestSuite.model_validate(
        {
            "suite_name": "Suite",
            "target_prompt_path": "unused.txt",
            "test_cases": [
                {
                    "id": "TC_002",
                    "description": "Low quality output",
                    "input_variables": {"user_query": "build an endpoint"},
                    "rubric": {
                        "metric": "technical_accuracy",
                        "criteria": "Is the output high quality?",
                        "passing_threshold": 4,
                    },
                }
            ],
        }
    )
    template_text = "Task: ${user_query}"

    client = Mock()
    client.models.generate_content.side_effect = [
        _fake_generate_content_response("mediocre output"),
        SimpleNamespace(parsed=JudgeVerdict(score=2, reasoning="Missing key details.")),
    ]

    results = build_results(suite, template_text, client)

    assert results[0].status == "FAIL"
    assert results[0].judge_verdict.score == 2


def test_build_results_skips_judge_when_no_rubric():
    suite = TestSuite.model_validate(
        {
            "suite_name": "Suite",
            "target_prompt_path": "unused.txt",
            "test_cases": [
                {
                    "id": "TC_003",
                    "description": "No rubric",
                    "input_variables": {"user_query": "build an endpoint"},
                }
            ],
        }
    )
    template_text = "Task: ${user_query}"

    client = Mock()
    client.models.generate_content.return_value = _fake_generate_content_response("some output")

    results = build_results(suite, template_text, client)

    assert results[0].judge_verdict is None
    assert results[0].status == "PASS"
    assert client.models.generate_content.call_count == 1


def test_build_results_records_failure_when_model_returns_none_text():
    suite = TestSuite.model_validate(
        {
            "suite_name": "Suite",
            "target_prompt_path": "unused.txt",
            "test_cases": [
                {
                    "id": "TC_004",
                    "description": "Safety-blocked response",
                    "input_variables": {"user_query": "build an endpoint"},
                    "assertions": {
                        "expected_format": "json",
                    },
                }
            ],
        }
    )
    template_text = "Task: ${user_query}"

    client = Mock()
    client.models.generate_content.return_value = _fake_generate_content_response(None)

    results = build_results(suite, template_text, client)

    assert len(results) == 1
    assert results[0].status == "FAIL"
    assert results[0].deterministic.failures


def test_build_results_records_failure_when_judge_returns_no_verdict():
    suite = TestSuite.model_validate(
        {
            "suite_name": "Suite",
            "target_prompt_path": "unused.txt",
            "test_cases": [
                {
                    "id": "TC_005",
                    "description": "Judge fails to parse",
                    "input_variables": {"user_query": "build an endpoint"},
                    "rubric": {
                        "metric": "technical_accuracy",
                        "criteria": "Is the output high quality?",
                        "passing_threshold": 4,
                    },
                }
            ],
        }
    )
    template_text = "Task: ${user_query}"

    client = Mock()
    client.models.generate_content.side_effect = [
        _fake_generate_content_response("some output"),
        SimpleNamespace(parsed=None),
    ]

    results = build_results(suite, template_text, client)

    assert len(results) == 1
    assert results[0].status == "FAIL"
    assert results[0].deterministic.failures


def _write_suite(tmp_path, rubric=None):
    (tmp_path / "prompt.txt").write_text("Task: ${user_query}")
    case = {"id": "TC_001", "description": "d", "input_variables": {"user_query": "q"}}
    if rubric:
        case["rubric"] = rubric
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        json.dumps(
            {"suite_name": "Suite", "target_prompt_path": "prompt.txt", "test_cases": [case]}
        )
    )
    return suite_path


def test_main_returns_2_when_suite_file_is_missing(tmp_path, capsys):
    assert main([str(tmp_path / "nope.json")]) == 2
    assert "could not load suite" in capsys.readouterr().err


def test_main_returns_2_when_api_key_is_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    suite_path = _write_suite(tmp_path)

    assert main([str(suite_path)]) == 2
    assert "GEMINI_API_KEY" in capsys.readouterr().err


def test_main_resolves_prompt_relative_to_suite_and_returns_0_on_pass(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    monkeypatch.chdir(tmp_path.parent)
    suite_path = _write_suite(tmp_path)
    client = Mock()
    client.models.generate_content.return_value = _fake_generate_content_response("out")

    with patch("judgedread.cli.genai.Client", return_value=client):
        code = main([str(suite_path), "--history", str(tmp_path / "h.jsonl")])

    assert code == 0
    assert client.models.generate_content.call_args.kwargs["contents"] == "Task: q"


def test_main_returns_1_when_any_case_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    suite_path = _write_suite(
        tmp_path, rubric={"metric": "m", "criteria": "c", "passing_threshold": 4}
    )
    client = Mock()
    client.models.generate_content.side_effect = [
        _fake_generate_content_response("out"),
        SimpleNamespace(parsed=JudgeVerdict(score=2, reasoning="weak")),
    ]

    with patch("judgedread.cli.genai.Client", return_value=client):
        code = main([str(suite_path), "--history", str(tmp_path / "h.jsonl")])

    assert code == 1


def test_build_results_uses_given_model_for_prompt_and_judge():
    suite = TestSuite.model_validate(
        {
            "suite_name": "Suite",
            "target_prompt_path": "unused.txt",
            "test_cases": [
                {
                    "id": "TC_006",
                    "description": "Custom model",
                    "input_variables": {"user_query": "build an endpoint"},
                    "rubric": {"metric": "m", "criteria": "c", "passing_threshold": 4},
                }
            ],
        }
    )
    client = Mock()
    client.models.generate_content.side_effect = [
        _fake_generate_content_response("out"),
        SimpleNamespace(parsed=JudgeVerdict(score=5, reasoning="ok")),
    ]

    build_results(suite, "Task: ${user_query}", client, model="gemma3:4b")

    models = [c.kwargs["model"] for c in client.models.generate_content.call_args_list]
    assert models == ["gemma3:4b", "gemma3:4b"]


def test_main_local_runs_on_ollama_without_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    suite_path = _write_suite(tmp_path)
    client = Mock()
    client.models.generate_content.return_value = _fake_generate_content_response("out")

    with (
        patch("judgedread.cli.OllamaClient", return_value=client),
        patch("judgedread.cli.genai.Client") as gemini,
    ):
        code = main([str(suite_path), "--local", "--history", str(tmp_path / "h.jsonl")])

    assert code == 0
    gemini.assert_not_called()
    client.check_model.assert_called_once_with("gemma3:4b")
    assert client.models.generate_content.call_args.kwargs["model"] == "gemma3:4b"


def test_main_local_model_overrides_default_model(tmp_path):
    suite_path = _write_suite(tmp_path)
    client = Mock()
    client.models.generate_content.return_value = _fake_generate_content_response("out")

    with patch("judgedread.cli.OllamaClient", return_value=client):
        main(
            [
                str(suite_path),
                "--local",
                "--local-model",
                "mistral",
                "--history",
                str(tmp_path / "h.jsonl"),
            ]
        )

    client.check_model.assert_called_once_with("mistral")
    assert client.models.generate_content.call_args.kwargs["model"] == "mistral"


def test_main_local_returns_2_when_ollama_is_unavailable(tmp_path, capsys):
    suite_path = _write_suite(tmp_path)
    client = Mock()
    client.check_model.side_effect = OllamaError("Ollama is not reachable")

    with patch("judgedread.cli.OllamaClient", return_value=client):
        code = main([str(suite_path), "--local"])

    assert code == 2
    assert "Ollama is not reachable" in capsys.readouterr().err
    client.models.generate_content.assert_not_called()
