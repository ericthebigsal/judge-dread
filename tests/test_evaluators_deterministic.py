from judgedread.evaluators import evaluate_deterministic
from judgedread.models import Assertions


def test_valid_json_with_all_required_keys_passes():
    assertions = Assertions(
        expected_format="json",
        required_keys=["path", "method"],
        banned_phrases=["ERROR"],
    )
    output = '{"path": "/profile", "method": "POST"}'

    result = evaluate_deterministic(output, assertions)

    assert result.passed is True
    assert result.failures == []
    assert result.parsed_json == {"path": "/profile", "method": "POST"}


def test_invalid_json_fails_immediately():
    assertions = Assertions(expected_format="json")
    output = "not valid json {"

    result = evaluate_deterministic(output, assertions)

    assert result.passed is False
    assert any("invalid JSON" in f for f in result.failures)
    assert result.parsed_json is None


def test_missing_required_keys_fails():
    assertions = Assertions(expected_format="json", required_keys=["path", "parameters"])
    output = '{"path": "/profile"}'

    result = evaluate_deterministic(output, assertions)

    assert result.passed is False
    assert any("parameters" in f for f in result.failures)


def test_banned_phrase_fails_case_insensitively():
    assertions = Assertions(banned_phrases=["unsupported"])
    output = "This request is Unsupported by the API."

    result = evaluate_deterministic(output, assertions)

    assert result.passed is False
    assert any("unsupported" in f for f in result.failures)


def test_no_assertions_configured_always_passes():
    assertions = Assertions()
    result = evaluate_deterministic("anything at all", assertions)
    assert result.passed is True
    assert result.failures == []
