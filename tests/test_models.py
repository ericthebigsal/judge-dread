import pytest
from pydantic import ValidationError

from judgedread.models import (
    Assertions,
    DeterministicResult,
    JudgeVerdict,
    Rubric,
    TestCase,
    TestCaseResult,
    TestSuite,
)


def test_parses_full_test_suite():
    data = {
        "suite_name": "API Spec Generator Evaluator",
        "target_prompt_path": "prompts/system_prompt.txt",
        "test_cases": [
            {
                "id": "TC_001",
                "description": "Verify valid JSON output and parameter parsing",
                "input_variables": {
                    "user_query": "Build a profile creation endpoint mapping email and phone."
                },
                "assertions": {
                    "expected_format": "json",
                    "required_keys": ["path", "method", "parameters"],
                    "banned_phrases": ["ERROR", "SOAP", "unsupported"],
                },
                "rubric": {
                    "metric": "technical_accuracy",
                    "criteria": "Does the generated payload schema correctly require string formats for email and phone numbers?",
                    "passing_threshold": 4,
                },
            }
        ],
    }

    suite = TestSuite.model_validate(data)

    assert suite.suite_name == "API Spec Generator Evaluator"
    assert len(suite.test_cases) == 1
    tc = suite.test_cases[0]
    assert tc.id == "TC_001"
    assert tc.input_variables["user_query"].startswith("Build a profile")
    assert tc.assertions.expected_format == "json"
    assert tc.assertions.required_keys == ["path", "method", "parameters"]
    assert tc.rubric.passing_threshold == 4


def test_test_case_without_rubric_defaults_to_none():
    tc = TestCase.model_validate(
        {
            "id": "TC_002",
            "description": "No rubric case",
            "input_variables": {},
        }
    )
    assert tc.rubric is None
    assert tc.assertions.required_keys == []
    assert tc.assertions.banned_phrases == []


def test_test_suite_requires_test_cases_field():
    with pytest.raises(ValidationError):
        TestSuite.model_validate({"suite_name": "x", "target_prompt_path": "y"})


def test_result_models_hold_expected_shapes():
    det = DeterministicResult(passed=True, failures=[], parsed_json={"a": 1})
    verdict = JudgeVerdict(score=5, reasoning="Meets all criteria.")
    result = TestCaseResult(
        id="TC_001",
        description="desc",
        status="PASS",
        latency_ms=123.4,
        input_tokens=10,
        output_tokens=20,
        deterministic=det,
        judge_verdict=verdict,
        raw_output="{}",
    )
    assert result.status == "PASS"
    assert result.judge_verdict.score == 5
