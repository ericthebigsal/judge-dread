# Project Proposal: "JudgeDread" – An Automated AI Evals & Prompt Regression Framework

---

## Part 1: Human-Readable Overview (For GitHub & Founders)

### Executive Summary

When building AI features, tweaking a single system prompt to fix a niche edge case can silently degrade performance across ten other scenarios. **JudgeDread** is a lightweight, local-first evaluation (Evals) framework built from scratch in Python to prevent prompt regression. It allows product builders to define strict testing datasets, execute programmatic assertions against AI outputs, and use an LLM-as-a-judge to track semantic accuracy, formatting compliance, and latency metrics before code hits production.

### The Problem Space

Startups and post-startup engineering teams are shipping AI features rapidly, but testing remains overwhelmingly manual, anecdotal, or non-existent ("looks good to me" testing). Relying on manual checks slows down shipping velocity, while completely unguided prompt updates risk introducing catastrophic silent regressions in user-facing applications.

### Core Features & Architecture

* **Deterministic Assertions:** Raw Python evaluation steps verifying schema enforcement (e.g., ensuring output is valid JSON matching a specific structural schema, or checking for banned terms/PII leaks).
* **Semantic "LLM-as-a-Judge" Evaluators:** Graded rubrics passed to a fast, local, or edge LLM (like `gemini-2.5-flash`) to measure complex characteristics like tone, correctness against a ground-truth dataset, or compliance with technical constraints.
* **Regression Tracking & Diffing:** A terminal-based testing harness that logs performance history, calculates passing rates, and highlights visual diffs when a prompt change causes a previously passing test to fail.
* **Cost & Latency Tracking:** Monitors token consumption estimates and API latency across test runs to ensure prompt changes haven't introduced expensive performance bottlenecks.

### Why This Fits a "Product Builder" Role

This project explicitly targets the bridge between **product quality assurance** and **backend engineering precision**. It demonstrates an elite understanding of the core operational reality of production AI: managing non-deterministic software with data-driven guardrails and regression suites.

---

## Part 2: Technical Specification (Tailored for LLM Context Windows)

```json
{
  "system_context": {
    "project_name": "judgedread",
    "target_environment": "Terminal / Python 3.10+",
    "primary_dependencies": ["numpy", "pydantic", "google-genai", "rich"],
    "architecture_style": "Functional, configuration-driven, strictly framework-less"
  }
}

```

### LLM Instructions: System Design & Schema Requirements

Act as an expert AI Platform Engineer. We are building a lightweight AI evaluation suite completely from scratch. Do not use generic abstraction libraries like LangChain, Phoenix, or Ragas. Implement the backend engine using raw Python, Pydantic for validation, and the standard Google GenAI SDK.

#### 1. Configuration & Test Dataset Schema

Every evaluation suite runs against a declared test matrix file (`test_suite.json`). The JSON schema must strictly conform to this structure:

```json
{
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

#### 2. The Execution Engine (`engine.py`)

The engine must execute the following pipeline chronologically:

1. **Load State:** Read `test_suite.json` and load the string target prompt from the specified text file path.
2. **Hydrate & Invoke:** For each test case, substitute the `input_variables` into the target prompt template. Execute the text generation call to Gemini using `gemini-2.5-flash`.
3. **Benchmark:** Wrap the API call with timers to capture raw processing duration. Read metadata output to track input/output token metrics.

#### 3. Assertion & Judge Pipelines (`evaluators.py`)

Implement two distinct assessment pipelines:

* **Deterministic Evaluation:**
* If `expected_format` is `"json"`, attempt to load the model's output using `json.loads()`. Fail the test case immediately if it encounters a parsing failure.
* Verify the existence of all listed `required_keys`.
* Iterate through `banned_phrases` and flag errors if regex matches are true.


* **LLM-as-a-Judge Evaluation (Semantic Grading):**
* For test cases containing a `rubric` block, generate an internal judge call using a rigid structure. The system context for the judge should be:


```text
You are an objective, strict QA Code Judge. Rate the following generated AI output based on the provided Criteria. Provide a numerical score between 1 and 5, followed by a one-sentence reasoning explanation.

CRITERIA: {criteria}
AI OUTPUT TO EVALUATE: {output}

Your response MUST exactly match this JSON schema:
{
  "score": int,
  "reasoning": "string"
}

```


* Use Structured Outputs with Pydantic to guarantee the judge response conforms to the schema. If the parsed integer `score` is less than the `passing_threshold`, mark the semantic evaluation step as failed.



#### 4. Terminal Reporter UI (`report.py`)

* Use the `rich` Python library to format final execution outputs cleanly inside the terminal.
* Render a dense summary table capturing: Test ID, Status (PASS/FAIL), Processing Latency (ms), Tokens Consumed, Deterministic Score, and Judge Score.
* If a historical test log exists (`.eval_history.jsonl`), compare the pass rate of the current run against the preceding run. Output a clear visual notification if performance metrics have regressed.

---

### Suggested First Prompt to pass to your AI assistant:

> "Let's start building the JudgeDread framework specified in the specification layout. Please write out `engine.py` and the Pydantic data models required to parse the `test_suite.json` schema cleanly. Keep it strictly Python-native without adding unneeded third-party libraries."
