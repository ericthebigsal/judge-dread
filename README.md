# JudgeDread

[![Tests](https://github.com/ericthebigsal/judge-dread/actions/workflows/tests.yml/badge.svg)](https://github.com/ericthebigsal/judge-dread/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

A local-first, framework-less Python harness for catching AI prompt regressions before they ship — built entirely on Gemini's free tier, so running it costs nothing but rate-limit patience.

## Elevator pitch

Every time you tweak a system prompt to fix one broken case, you risk quietly breaking ten others — and "looks good to me" is not a test suite. JudgeDread turns your prompts into something you can actually regression-test: define a suite of cases in JSON, run them against your prompt, check the output two ways — hard deterministic assertions (valid JSON? required fields present? banned phrases absent?) and a second AI acting as an impartial judge scoring semantic quality against a rubric — then get a clean terminal report that calls out, by name, any test that used to pass and now doesn't. No LangChain, no eval platform, no billing account. Just Python, Pydantic, and a free Gemini API key.

## Why this exists

Teams shipping AI features usually test prompts by eyeballing a few outputs and shipping if they look fine. That doesn't scale, and it doesn't catch regressions — a prompt edit that fixes edge case A can silently break edge cases B through J with nobody noticing until a user does. JudgeDread is the smallest possible tool that closes that gap: a real, versionable test suite for prompts, with both mechanical checks (fast, free, deterministic) and semantic checks (an LLM judging quality against a written rubric), plus a history file so "did this get worse?" is a computed answer, not a guess.

## Features

- **Deterministic assertions** — JSON validity, required keys present, banned phrases absent. Zero API cost, instant.
- **LLM-as-judge scoring** — a second model call grades the output against a written rubric (1–5) using Gemini's structured-output mode, so the verdict is a validated `{score, reasoning}` object, never hand-parsed text.
- **Regression tracking** — every run appends to `.eval_history.jsonl`; the next run diffs against the previous one and calls out, by test ID, any case that flipped from PASS to FAIL.
- **Rich terminal report** — a clean table with status, latency, token usage, and both check results per test case.
- **Zero billing risk** — runs on `gemini-2.5-flash` on the free tier. Once you hit the rate limit, calls fail with a 429; there is no path to an unexpected charge.
- **Fully testable offline** — every one of the 26 tests in this repo mocks the Gemini client. You can clone this repo and run the test suite with no API key at all.

## Quick start

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Get a free API key (no credit card required)
#    https://aistudio.google.com/apikey
export GEMINI_API_KEY="AIza..."

# 3. Run the example suite
python -m judgedread.cli examples/test_suite.json
```

You should see a Rich table with one row (`TC_001`), a pass-rate line, and a new `.eval_history.jsonl` file recording the run.

## Writing your own test suite

A test suite is a JSON file pointing at a prompt template plus a list of test cases:

```json
{
  "suite_name": "My Prompt Evaluator",
  "target_prompt_path": "prompts/system_prompt.txt",
  "test_cases": [
    {
      "id": "TC_001",
      "description": "What this case checks",
      "input_variables": { "user_query": "..." },
      "assertions": {
        "expected_format": "json",
        "required_keys": ["path", "method"],
        "banned_phrases": ["ERROR", "unsupported"]
      },
      "rubric": {
        "metric": "technical_accuracy",
        "criteria": "Does the output correctly do X?",
        "passing_threshold": 4
      }
    }
  ]
}
```

The prompt template file uses `${variable}` placeholders (Python's `string.Template` syntax), substituted from each test case's `input_variables` before the call is made. See `examples/system_prompt.txt` and `examples/test_suite.json` for a working reference.

`assertions` and `rubric` are both optional — a test case can run deterministic checks only, judge checks only, both, or (if you just want latency/token numbers) neither.

## Project structure

```
judgedread/
  models.py       # Pydantic schema: TestSuite, TestCase, Assertions, Rubric, results
  engine.py        # prompt hydration + Gemini invocation + timing/token capture
  evaluators.py     # deterministic checks + LLM-as-judge
  report.py       # Rich table rendering + history/regression tracking
  cli.py          # wiring: load suite -> run -> evaluate -> report
tests/            # 26 tests, all mocking the Gemini client — no API key needed
examples/         # a working prompt + test suite you can run immediately
```

## Testing

```bash
pytest -v
```

All 26 tests pass with zero network access and zero API key — the CLI, engine, and evaluator tests all inject a mocked client. The only thing that touches the real API is the manual smoke-test step above. This also means CI needs no secrets: [`.github/workflows/tests.yml`](.github/workflows/tests.yml) runs the full suite on every push and PR to `main`, on Python 3.10 and 3.12.

## License

MIT — see [`LICENSE`](LICENSE).

## Further reading

- [`docs/PRIMER.md`](docs/PRIMER.md) — a from-the-ground-up explainer on why AI prompt testing is hard and how JudgeDread's approach (deterministic + LLM-as-judge) addresses it, aimed at anyone with general programming background but no AI/ML background assumed.
- [`docs/BUILD_NOTES.md`](docs/BUILD_NOTES.md) — how this project was actually built: process, key decisions, and known rough edges.
- [`proposal.md`](proposal.md) — the original design proposal this project was built from.
