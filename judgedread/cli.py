import sys

from google import genai

from judgedread.engine import load_prompt_template, load_test_suite, run_test_case
from judgedread.evaluators import evaluate_deterministic, evaluate_judge
from judgedread.models import TestCaseResult, TestSuite
from judgedread.report import render_report


def build_results(suite: TestSuite, template_text: str, client) -> list[TestCaseResult]:
    results: list[TestCaseResult] = []

    for test_case in suite.test_cases:
        output, latency_ms, input_tokens, output_tokens = run_test_case(
            client, test_case, template_text
        )

        deterministic = evaluate_deterministic(output, test_case.assertions)

        judge_verdict = None
        semantic_passed = True
        if test_case.rubric is not None:
            judge_verdict = evaluate_judge(client, test_case.rubric, output)
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

    return results


def main(suite_path: str = "test_suite.json") -> None:
    suite = load_test_suite(suite_path)
    template_text = load_prompt_template(suite.target_prompt_path)
    client = genai.Client()
    results = build_results(suite, template_text, client)
    render_report(suite.suite_name, results)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test_suite.json")
