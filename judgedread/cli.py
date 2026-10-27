import argparse
import json
import os
import sys
from pathlib import Path

from google import genai
from pydantic import ValidationError

from judgedread.engine import load_prompt_template, load_test_suite, run_test_case
from judgedread.evaluators import evaluate_deterministic, evaluate_judge
from judgedread.models import DeterministicResult, TestCaseResult, TestSuite
from judgedread.report import DEFAULT_HISTORY_PATH, render_report


def build_results(suite: TestSuite, template_text: str, client) -> list[TestCaseResult]:
    results: list[TestCaseResult] = []

    for test_case in suite.test_cases:
        try:
            output, latency_ms, input_tokens, output_tokens = run_test_case(
                client, test_case, template_text
            )

            deterministic = evaluate_deterministic(output, test_case.assertions)

            judge_verdict = None
            semantic_passed = True
            if test_case.rubric is not None:
                judge_verdict = evaluate_judge(client, test_case.rubric, output)
                if judge_verdict is None:
                    raise RuntimeError("judge did not return a parsable verdict")
                semantic_passed = judge_verdict.score >= test_case.rubric.passing_threshold

            status = "PASS" if deterministic.passed and semantic_passed else "FAIL"

            results.append(
                TestCaseResult(
                    id=test_case.id,
                    description=test_case.description,
                    status=status,
                    latency_ms=latency_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    deterministic=deterministic,
                    judge_verdict=judge_verdict,
                    raw_output=output,
                )
            )
        except Exception as exc:
            results.append(
                TestCaseResult(
                    id=test_case.id,
                    description=test_case.description,
                    status="FAIL",
                    latency_ms=0.0,
                    input_tokens=0,
                    output_tokens=0,
                    deterministic=DeterministicResult(
                        passed=False, failures=[f"test case errored: {exc}"]
                    ),
                    judge_verdict=None,
                    raw_output="",
                )
            )

    return results


def main(argv: list[str] | None = None) -> int:
    """Run a suite and return a process exit code: 0 all passed, 1 any failed, 2 usage error."""
    parser = argparse.ArgumentParser(
        prog="judgedread", description="Run a JudgeDread prompt-evaluation suite."
    )
    parser.add_argument("suite", help="path to the test suite JSON file")
    parser.add_argument(
        "--history",
        type=Path,
        default=DEFAULT_HISTORY_PATH,
        help=f"run-history file used for regression tracking (default: {DEFAULT_HISTORY_PATH})",
    )
    args = parser.parse_args(argv)

    try:
        suite = load_test_suite(args.suite)
        # Prompt paths are relative to the suite file, so a suite runs from any directory.
        prompt_path = Path(args.suite).parent / suite.target_prompt_path
        template_text = load_prompt_template(prompt_path)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        print(f"judgedread: could not load suite: {exc}", file=sys.stderr)
        return 2

    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print(
            "judgedread: GEMINI_API_KEY is not set. "
            "Get a free key at https://aistudio.google.com/apikey",
            file=sys.stderr,
        )
        return 2

    client = genai.Client()
    results = build_results(suite, template_text, client)
    render_report(suite.suite_name, results, history_path=args.history)
    return 0 if all(r.status == "PASS" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
