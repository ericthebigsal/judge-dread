import json
import re

from judgedread.models import Assertions, DeterministicResult


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
