from types import SimpleNamespace
from unittest.mock import Mock

from judgedread.cli import build_results
from judgedread.models import JudgeVerdict, TestSuite


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
