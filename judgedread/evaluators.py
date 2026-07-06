import json
import re

from google.genai import types

from judgedread.models import Assertions, DeterministicResult, JudgeVerdict, Rubric


def evaluate_deterministic(output: str, assertions: Assertions) -> DeterministicResult:
    failures: list[str] = []
    parsed: dict | None = None

    if assertions.expected_format == "json":
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError as exc:
            failures.append(f"invalid JSON: {exc}")

    if assertions.required_keys:
        if parsed is None:
            failures.append("cannot check required_keys: output is not valid JSON")
        else:
            missing = [key for key in assertions.required_keys if key not in parsed]
            if missing:
                failures.append(f"missing required keys: {missing}")

    for phrase in assertions.banned_phrases:
        if re.search(re.escape(phrase), output, re.IGNORECASE):
            failures.append(f"banned phrase found: {phrase}")

    return DeterministicResult(passed=not failures, failures=failures, parsed_json=parsed)


JUDGE_MODEL = "gemini-2.5-flash"

JUDGE_PROMPT_TEMPLATE = (
    "You are an objective, strict QA Code Judge. Rate the following generated AI "
    "output based on the provided Criteria. Provide a numerical score between 1 "
    "and 5, followed by a one-sentence reasoning explanation.\n\n"
    "CRITERIA: {criteria}\n"
    "AI OUTPUT TO EVALUATE: {output}"
)


def evaluate_judge(
    client, rubric: Rubric, output: str, model: str = JUDGE_MODEL
) -> JudgeVerdict:
    prompt = JUDGE_PROMPT_TEMPLATE.format(criteria=rubric.criteria, output=output)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=JudgeVerdict,
        ),
    )
    return response.parsed
