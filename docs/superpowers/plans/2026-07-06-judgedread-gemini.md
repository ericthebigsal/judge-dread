# JudgeDread (Gemini Free-Tier Edition) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build JudgeDread, a local-first, framework-less Python evaluation harness that runs test suites against a Gemini system prompt on the **free tier** (zero cost, rate-limited rather than billed), checks outputs with deterministic assertions and an LLM-as-judge, and reports pass/fail with latency, token, and regression tracking in the terminal.

**Architecture:** A small package (`judgedread/`) with one file per responsibility — Pydantic schema, prompt hydration + Gemini invocation, evaluators (deterministic + judge), and a Rich terminal reporter with a JSONL history log — wired together by a thin CLI module. Every function that calls the Gemini API takes the client as a parameter, so all business logic is testable with a mocked client and no network access, and no API key is required to run the test suite.

**Tech Stack:** Python 3.10+, `google-genai` SDK, `pydantic` v2, `rich`, `pytest` for tests.

## Global Constraints

- Target environment: terminal / Python 3.10+.
- Dependencies limited to `google-genai`, `pydantic`, `rich` (plus `pytest` as a dev dependency). No LangChain, Phoenix, Ragas, or other agent/eval frameworks — raw Python only.
- Judge and target-prompt calls default to model `gemini-2.5-flash` on the **free tier** — no billing account required, calls fail with a rate-limit error rather than incurring cost. Every function that calls the model accepts a `model: str` override.
- API key comes from `GEMINI_API_KEY` (or `GOOGLE_API_KEY`), read automatically by `genai.Client()` — never hardcode a key.
- The judge must use `response_schema=JudgeVerdict` (a Pydantic model) on `GenerateContentConfig` and read `response.parsed` for structured output — never hand-parse the judge's raw text with `json.loads`.
- History file is `.eval_history.jsonl`, one JSON record per run, newline-delimited; regression check compares per-test-case status against the immediately preceding line and calls out any test that flipped from PASS to FAIL by ID.
- Every function that talks to the Gemini API takes the client as an explicit parameter (dependency injection) so it can be tested with a mock and no real API key.

---

## File Structure

```
JudgeDred/
  pyproject.toml
  judgedread/
    __init__.py
    models.py       # Pydantic schema: TestSuite, TestCase, Assertions, Rubric, results
    engine.py        # prompt hydration + Gemini invocation + timing/token capture
    evaluators.py     # deterministic checks + LLM-as-judge
    report.py       # Rich table rendering + history/regression tracking
    cli.py          # wiring: load suite -> run -> evaluate -> report
  tests/
    test_models.py
    test_evaluators_deterministic.py
    test_engine.py
    test_evaluators_judge.py
    test_report.py
    test_cli.py
  examples/
    system_prompt.txt
    test_suite.json
```

---

### Task 1: Project scaffolding and Pydantic schema

**Files:**
- Create: `pyproject.toml`
- Create: `judgedread/__init__.py`
- Create: `judgedread/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces: `TestSuite`, `TestCase`, `Assertions`, `Rubric` (parsed from `test_suite.json`), `DeterministicResult`, `JudgeVerdict`, `TestCaseResult` (used by every later task).

- [ ] **Step 1: Initialize git and create the project layout**

```bash
cd /Users/ericsalerno/Documents/JudgeDred
git init
mkdir -p judgedread tests examples
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "judgedread"
version = "0.1.0"
description = "Local-first AI evals and prompt regression framework, built on Gemini's free tier"
requires-python = ">=3.10"
dependencies = [
    "google-genai>=0.3.0",
    "pydantic>=2.0",
    "rich>=13.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["judgedread"]
```

- [ ] **Step 3: Create `.gitignore`**

```
__pycache__/
*.pyc
.venv/
.eval_history.jsonl
*.egg-info/
```

- [ ] **Step 4: Create empty package marker**

```bash
touch judgedread/__init__.py
```

- [ ] **Step 5: Write the failing test for the schema**

`tests/test_models.py`:

```python
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
```

- [ ] **Step 6: Run the test to verify it fails**

Run: `pip install -e ".[dev]" && pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'judgedread.models'`

- [ ] **Step 7: Write `judgedread/models.py`**

```python
from typing import Optional

from pydantic import BaseModel, Field


class Assertions(BaseModel):
    expected_format: Optional[str] = None
    required_keys: list[str] = Field(default_factory=list)
    banned_phrases: list[str] = Field(default_factory=list)


class Rubric(BaseModel):
    metric: str
    criteria: str
    passing_threshold: int


class TestCase(BaseModel):
    id: str
    description: str
    input_variables: dict[str, str] = Field(default_factory=dict)
    assertions: Assertions = Field(default_factory=Assertions)
    rubric: Optional[Rubric] = None


class TestSuite(BaseModel):
    suite_name: str
    target_prompt_path: str
    test_cases: list[TestCase]


class DeterministicResult(BaseModel):
    passed: bool
    failures: list[str] = Field(default_factory=list)
    parsed_json: Optional[dict] = None


class JudgeVerdict(BaseModel):
    score: int
    reasoning: str


class TestCaseResult(BaseModel):
    id: str
    description: str
    status: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    deterministic: DeterministicResult
    judge_verdict: Optional[JudgeVerdict] = None
    raw_output: str
```

- [ ] **Step 8: Run the test to verify it passes**

Run: `pytest tests/test_models.py -v`
Expected: PASS (4 tests)

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml .gitignore judgedread/__init__.py judgedread/models.py tests/test_models.py
git commit -m "feat: scaffold project and add Pydantic schema for test suites and results"
```

---

### Task 2: Deterministic evaluator

**Files:**
- Create: `judgedread/evaluators.py`
- Test: `tests/test_evaluators_deterministic.py`

**Interfaces:**
- Consumes: `Assertions`, `DeterministicResult` from `judgedread.models` (Task 1).
- Produces: `evaluate_deterministic(output: str, assertions: Assertions) -> DeterministicResult`, used by Task 6 (CLI wiring).

- [ ] **Step 1: Write the failing tests**

`tests/test_evaluators_deterministic.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_evaluators_deterministic.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'judgedread.evaluators'`

- [ ] **Step 3: Write `judgedread/evaluators.py` (deterministic portion only)**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_evaluators_deterministic.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add judgedread/evaluators.py tests/test_evaluators_deterministic.py
git commit -m "feat: add deterministic evaluator for JSON, required keys, banned phrases"
```

---

### Task 3: Prompt hydration and Gemini invocation

**Files:**
- Create: `judgedread/engine.py`
- Test: `tests/test_engine.py`

**Interfaces:**
- Consumes: `TestSuite`, `TestCase` from `judgedread.models` (Task 1).
- Produces: `load_test_suite(path: str) -> TestSuite`, `load_prompt_template(path: str) -> str`, `hydrate_prompt(template_text: str, input_variables: dict[str, str]) -> str`, `invoke_model(client, prompt: str, model: str = "gemini-2.5-flash") -> tuple[str, float, int, int]` (text, latency_ms, input_tokens, output_tokens), `run_test_case(client, test_case: TestCase, template_text: str, model: str = "gemini-2.5-flash") -> tuple[str, float, int, int]`. Used by Task 6 (CLI wiring).

- [ ] **Step 1: Write the failing tests**

`tests/test_engine.py`:

```python
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from judgedread.engine import (
    hydrate_prompt,
    invoke_model,
    load_prompt_template,
    load_test_suite,
    run_test_case,
)
from judgedread.models import TestCase


def _fake_response(text: str, input_tokens: int = 12, output_tokens: int = 34):
    return SimpleNamespace(
        text=text,
        usage_metadata=SimpleNamespace(
            prompt_token_count=input_tokens, candidates_token_count=output_tokens
        ),
    )


def test_load_test_suite_parses_json_file(tmp_path):
    suite_path = tmp_path / "test_suite.json"
    suite_path.write_text(
        json.dumps(
            {
                "suite_name": "Suite",
                "target_prompt_path": "prompts/system_prompt.txt",
                "test_cases": [
                    {"id": "TC_001", "description": "d", "input_variables": {}}
                ],
            }
        )
    )

    suite = load_test_suite(str(suite_path))

    assert suite.suite_name == "Suite"
    assert suite.test_cases[0].id == "TC_001"


def test_load_prompt_template_reads_file_text(tmp_path):
    prompt_path = tmp_path / "system_prompt.txt"
    prompt_path.write_text("Respond to: ${user_query}")

    text = load_prompt_template(str(prompt_path))

    assert text == "Respond to: ${user_query}"


def test_hydrate_prompt_substitutes_variables():
    template = "Build an endpoint for: ${user_query}. Format: ${fmt}"
    result = hydrate_prompt(template, {"user_query": "profile creation", "fmt": "json"})
    assert result == "Build an endpoint for: profile creation. Format: json"


def test_hydrate_prompt_raises_on_missing_variable():
    template = "Needs ${missing_var}"
    with pytest.raises(KeyError):
        hydrate_prompt(template, {})


def test_invoke_model_returns_text_latency_and_token_counts():
    client = Mock()
    client.models.generate_content.return_value = _fake_response(
        '{"path": "/profile"}', input_tokens=50, output_tokens=75
    )

    text, latency_ms, input_tokens, output_tokens = invoke_model(
        client, "some prompt", model="gemini-2.5-flash"
    )

    assert text == '{"path": "/profile"}'
    assert latency_ms >= 0
    assert input_tokens == 50
    assert output_tokens == 75
    client.models.generate_content.assert_called_once()
    _, kwargs = client.models.generate_content.call_args
    assert kwargs["model"] == "gemini-2.5-flash"
    assert kwargs["contents"] == "some prompt"


def test_run_test_case_hydrates_then_invokes():
    client = Mock()
    client.models.generate_content.return_value = _fake_response("output text")
    test_case = TestCase.model_validate(
        {
            "id": "TC_001",
            "description": "d",
            "input_variables": {"user_query": "build a thing"},
        }
    )
    template_text = "Task: ${user_query}"

    text, latency_ms, input_tokens, output_tokens = run_test_case(
        client, test_case, template_text
    )

    assert text == "output text"
    _, kwargs = client.models.generate_content.call_args
    assert kwargs["contents"] == "Task: build a thing"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'judgedread.engine'`

- [ ] **Step 3: Write `judgedread/engine.py`**

```python
import json
import time
from pathlib import Path
from string import Template

from google.genai import types

from judgedread.models import TestCase, TestSuite

DEFAULT_MODEL = "gemini-2.5-flash"


def load_test_suite(path: str) -> TestSuite:
    data = json.loads(Path(path).read_text())
    return TestSuite.model_validate(data)


def load_prompt_template(path: str) -> str:
    return Path(path).read_text()


def hydrate_prompt(template_text: str, input_variables: dict[str, str]) -> str:
    return Template(template_text).substitute(**input_variables)


def invoke_model(
    client, prompt: str, model: str = DEFAULT_MODEL
) -> tuple[str, float, int, int]:
    start = time.perf_counter()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(max_output_tokens=2048),
    )
    latency_ms = (time.perf_counter() - start) * 1000
    return (
        response.text,
        latency_ms,
        response.usage_metadata.prompt_token_count,
        response.usage_metadata.candidates_token_count,
    )


def run_test_case(
    client, test_case: TestCase, template_text: str, model: str = DEFAULT_MODEL
) -> tuple[str, float, int, int]:
    prompt = hydrate_prompt(template_text, test_case.input_variables)
    return invoke_model(client, prompt, model=model)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_engine.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add judgedread/engine.py tests/test_engine.py
git commit -m "feat: add prompt hydration and Gemini invocation with latency/token capture"
```

---

### Task 4: LLM-as-judge evaluator

**Files:**
- Modify: `judgedread/evaluators.py` (add judge function)
- Test: `tests/test_evaluators_judge.py`

**Interfaces:**
- Consumes: `Rubric`, `JudgeVerdict` from `judgedread.models` (Task 1).
- Produces: `evaluate_judge(client, rubric: Rubric, output: str, model: str = "gemini-2.5-flash") -> JudgeVerdict`. Used by Task 6 (CLI wiring).

- [ ] **Step 1: Write the failing test**

`tests/test_evaluators_judge.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evaluators_judge.py -v`
Expected: FAIL with `ImportError: cannot import name 'evaluate_judge' from 'judgedread.evaluators'`

- [ ] **Step 3: Add the judge function to `judgedread/evaluators.py`**

Append to the existing `judgedread/evaluators.py` (keep the Task 2 imports and `evaluate_deterministic` function untouched):

```python
from google.genai import types

from judgedread.models import JudgeVerdict, Rubric

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
```

The top of `judgedread/evaluators.py` should now have these imports:

```python
import json
import re

from google.genai import types

from judgedread.models import Assertions, DeterministicResult, JudgeVerdict, Rubric
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_evaluators_judge.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Run the full evaluator test file together**

Run: `pytest tests/test_evaluators_deterministic.py tests/test_evaluators_judge.py -v`
Expected: PASS (6 tests total)

- [ ] **Step 6: Commit**

```bash
git add judgedread/evaluators.py tests/test_evaluators_judge.py
git commit -m "feat: add LLM-as-judge evaluator using Gemini structured outputs"
```

---

### Task 5: Rich terminal reporter with history and regression tracking

**Files:**
- Create: `judgedread/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `TestCaseResult` from `judgedread.models` (Task 1).
- Produces: `render_report(suite_name: str, results: list[TestCaseResult], history_path: Path = Path(".eval_history.jsonl")) -> float` (returns the computed pass rate). Used by Task 6 (CLI wiring).

- [ ] **Step 1: Write the failing tests**

`tests/test_report.py`:

```python
import json

from judgedread.models import DeterministicResult, JudgeVerdict, TestCaseResult
from judgedread.report import render_report


def _result(id_, status, judge_score=None):
    verdict = JudgeVerdict(score=judge_score, reasoning="r") if judge_score is not None else None
    return TestCaseResult(
        id=id_,
        description="desc",
        status=status,
        latency_ms=100.0,
        input_tokens=10,
        output_tokens=20,
        deterministic=DeterministicResult(passed=(status == "PASS")),
        judge_verdict=verdict,
        raw_output="{}",
    )


def test_render_report_writes_history_record_with_per_test_statuses(tmp_path, capsys):
    history_path = tmp_path / ".eval_history.jsonl"
    results = [_result("TC_001", "PASS"), _result("TC_002", "FAIL")]

    pass_rate = render_report("Suite", results, history_path=history_path)

    assert pass_rate == 0.5
    lines = history_path.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["suite_name"] == "Suite"
    assert record["pass_rate"] == 0.5
    assert record["results"] == [
        {"id": "TC_001", "status": "PASS"},
        {"id": "TC_002", "status": "FAIL"},
    ]
    captured = capsys.readouterr()
    assert "TC_001" in captured.out
    assert "TC_002" in captured.out


def test_render_report_appends_to_existing_history(tmp_path):
    history_path = tmp_path / ".eval_history.jsonl"
    history_path.write_text(
        json.dumps(
            {
                "timestamp": 1,
                "suite_name": "Suite",
                "pass_rate": 1.0,
                "results": [{"id": "TC_001", "status": "PASS"}, {"id": "TC_002", "status": "PASS"}],
            }
        )
        + "\n"
    )
    results = [_result("TC_001", "PASS"), _result("TC_002", "FAIL")]

    render_report("Suite", results, history_path=history_path)

    lines = history_path.read_text().strip().splitlines()
    assert len(lines) == 2


def test_render_report_flags_specific_test_that_flipped_to_failing(tmp_path, capsys):
    history_path = tmp_path / ".eval_history.jsonl"
    history_path.write_text(
        json.dumps(
            {
                "timestamp": 1,
                "suite_name": "Suite",
                "pass_rate": 1.0,
                "results": [{"id": "TC_001", "status": "PASS"}, {"id": "TC_002", "status": "PASS"}],
            }
        )
        + "\n"
    )
    results = [_result("TC_001", "PASS"), _result("TC_002", "FAIL")]

    render_report("Suite", results, history_path=history_path)

    captured = capsys.readouterr()
    assert "REGRESSION" in captured.out
    assert "TC_002" in captured.out
    assert "TC_001" not in captured.out.split("REGRESSION")[1]


def test_render_report_flags_aggregate_drop_when_test_ids_are_new(tmp_path, capsys):
    history_path = tmp_path / ".eval_history.jsonl"
    history_path.write_text(
        json.dumps(
            {
                "timestamp": 1,
                "suite_name": "Suite",
                "pass_rate": 1.0,
                "results": [{"id": "TC_OLD", "status": "PASS"}],
            }
        )
        + "\n"
    )
    results = [_result("TC_NEW_1", "PASS"), _result("TC_NEW_2", "FAIL")]

    render_report("Suite", results, history_path=history_path)

    captured = capsys.readouterr()
    assert "REGRESSION" in captured.out
    assert "pass rate dropped" in captured.out


def test_render_report_with_no_results_has_zero_pass_rate(tmp_path):
    history_path = tmp_path / ".eval_history.jsonl"
    pass_rate = render_report("Empty Suite", [], history_path=history_path)
    assert pass_rate == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'judgedread.report'`

- [ ] **Step 3: Write `judgedread/report.py`**

```python
import json
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

from judgedread.models import TestCaseResult

DEFAULT_HISTORY_PATH = Path(".eval_history.jsonl")


def render_report(
    suite_name: str,
    results: list[TestCaseResult],
    history_path: Path = DEFAULT_HISTORY_PATH,
) -> float:
    console = Console()
    table = Table(title=suite_name)
    table.add_column("Test ID")
    table.add_column("Status")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Deterministic")
    table.add_column("Judge Score")

    for result in results:
        deterministic_cell = (
            "PASS"
            if result.deterministic.passed
            else "FAIL: " + "; ".join(result.deterministic.failures)
        )
        judge_cell = f"{result.judge_verdict.score}/5" if result.judge_verdict else "-"
        table.add_row(
            result.id,
            "[green]PASS[/]" if result.status == "PASS" else "[red]FAIL[/]",
            f"{result.latency_ms:.0f}",
            str(result.input_tokens + result.output_tokens),
            deterministic_cell,
            judge_cell,
        )

    console.print(table)

    pass_rate = (
        sum(1 for r in results if r.status == "PASS") / len(results) if results else 0.0
    )
    console.print(f"Pass rate: {pass_rate:.0%}")

    _check_regression(pass_rate, results, history_path, console)
    _append_history(suite_name, pass_rate, results, history_path)

    return pass_rate


def _check_regression(
    pass_rate: float,
    results: list[TestCaseResult],
    history_path: Path,
    console: Console,
) -> None:
    if not history_path.exists():
        return
    lines = history_path.read_text().strip().splitlines()
    if not lines:
        return
    last_record = json.loads(lines[-1])
    last_statuses = {r["id"]: r["status"] for r in last_record.get("results", [])}

    flipped = [
        r.id for r in results if last_statuses.get(r.id) == "PASS" and r.status == "FAIL"
    ]
    if flipped:
        console.print(
            f"[bold red]REGRESSION: previously passing tests now failing: "
            f"{', '.join(flipped)}[/]"
        )
        return

    if pass_rate < last_record["pass_rate"]:
        console.print(
            f"[bold red]REGRESSION: pass rate dropped from "
            f"{last_record['pass_rate']:.0%} to {pass_rate:.0%}[/]"
        )


def _append_history(
    suite_name: str, pass_rate: float, results: list[TestCaseResult], history_path: Path
) -> None:
    record = {
        "timestamp": time.time(),
        "suite_name": suite_name,
        "pass_rate": pass_rate,
        "results": [{"id": r.id, "status": r.status} for r in results],
    }
    with open(history_path, "a") as f:
        f.write(json.dumps(record) + "\n")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_report.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add judgedread/report.py tests/test_report.py
git commit -m "feat: add Rich terminal reporter with JSONL history and regression detection"
```

---

### Task 6: CLI wiring and example fixtures

**Files:**
- Create: `judgedread/cli.py`
- Create: `examples/system_prompt.txt`
- Create: `examples/test_suite.json`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: everything from Tasks 1-5 (`TestSuite`, `TestCase`, `TestCaseResult` from `judgedread.models`; `load_test_suite`, `load_prompt_template`, `run_test_case` from `judgedread.engine`; `evaluate_deterministic`, `evaluate_judge` from `judgedread.evaluators`; `render_report` from `judgedread.report`).
- Produces: `build_results(suite: TestSuite, template_text: str, client) -> list[TestCaseResult]`, `main(suite_path: str = "test_suite.json") -> None`. This is the top-level entrypoint — no later task depends on it.

- [ ] **Step 1: Write the failing tests**

`tests/test_cli.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'judgedread.cli'`

- [ ] **Step 3: Write `judgedread/cli.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cli.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Create example fixtures for manual end-to-end runs**

`examples/system_prompt.txt`:

```
You are an API Spec Generator. Given a user request, output ONLY a JSON object
(no prose, no markdown fences) with this shape:
{"path": "<url path>", "method": "<HTTP verb>", "parameters": {"<field>": {"type": "string"}, ...}}

User request: ${user_query}
```

`examples/test_suite.json`:

```json
{
  "suite_name": "API Spec Generator Evaluator",
  "target_prompt_path": "examples/system_prompt.txt",
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
        "banned_phrases": ["ERROR", "SOAP", "unsupported"]
      },
      "rubric": {
        "metric": "technical_accuracy",
        "criteria": "Does the generated payload schema correctly require string formats for email and phone numbers?",
        "passing_threshold": 4
      }
    }
  ]
}
```

- [ ] **Step 6: Run the full test suite**

Run: `pytest -v`
Expected: PASS (24 tests total across all files)

- [ ] **Step 7: Commit**

```bash
git add judgedread/cli.py examples/system_prompt.txt examples/test_suite.json tests/test_cli.py
git commit -m "feat: wire CLI entrypoint and add example fixtures for manual end-to-end runs"
```

- [ ] **Step 8: (Manual, requires a free Gemini API key — not part of automated tests) Smoke test against the live API**

Get a free key from Google AI Studio (https://aistudio.google.com/apikey) — no credit card required for the free tier.

```bash
export GEMINI_API_KEY="AIza..."
python -m judgedread.cli examples/test_suite.json
```

Expected: a Rich table prints with one row (`TC_001`), a pass rate line, and `.eval_history.jsonl` is created in the working directory with one record.

---

## Post-Plan Notes

- Every task's tests run with zero network access and zero API key — the CLI test suite and the engine/evaluator tests all inject a `Mock()` client. Only Task 6 Step 8 touches the real API, and it's explicitly manual.
- **Cost model: $0.** `gemini-2.5-flash` on the free tier has no billing account attached — once you exceed the free tier's requests-per-minute or requests-per-day cap, calls return a rate-limit error (HTTP 429) instead of costing anything. There is no path to an unexpected charge on this setup.
- If free-tier rate limits are hit during a large suite run, `client.models.generate_content` will raise — consider adding a retry-with-backoff wrapper around `invoke_model` later if this becomes a problem in practice; it's intentionally left out of this plan (YAGNI) since the example suite is a single test case.
- The exact field names on `response.usage_metadata` (`prompt_token_count` / `candidates_token_count`) and the `response.parsed` structured-output behavior reflect the `google-genai` SDK as of this writing. If the live smoke test (Task 6 Step 8) shows a different shape, adjust `invoke_model` and `evaluate_judge` accordingly — the mocked tests will still pass either way since they assert against the shape this plan defines.
- If a rubric's `passing_threshold` or the judge's `score` scale ever needs to change from the fixed 1-5 range, that's a schema change in `judgedread/models.py` (`Rubric`, `JudgeVerdict`) and the `JUDGE_PROMPT_TEMPLATE` in `judgedread/evaluators.py` — both would need to move together.
