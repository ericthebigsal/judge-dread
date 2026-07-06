from types import SimpleNamespace
from unittest.mock import Mock

from judgedread.evaluators import evaluate_judge
from judgedread.models import JudgeVerdict, Rubric


def test_evaluate_judge_returns_parsed_verdict():
    client = Mock()
    verdict = JudgeVerdict(score=4, reasoning="Correctly enforces string formats.")
    client.models.generate_content.return_value = SimpleNamespace(parsed=verdict)
    rubric = Rubric(
        metric="technical_accuracy",
        criteria="Does the schema require string formats for email and phone?",
        passing_threshold=4,
    )

    result = evaluate_judge(client, rubric, '{"email": {"type": "string"}}')

    assert result == verdict
    client.models.generate_content.assert_called_once()
    _, kwargs = client.models.generate_content.call_args
    assert kwargs["model"] == "gemini-2.5-flash"
    assert kwargs["config"].response_schema is JudgeVerdict
    assert kwargs["config"].response_mime_type == "application/json"
    assert "Does the schema require string formats" in kwargs["contents"]
    assert '{"email": {"type": "string"}}' in kwargs["contents"]
