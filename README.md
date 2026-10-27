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
- **LLM-as-judge scoring** — a second model call grades the output against a written rubric (1–5) using Gemini's structured-output mode, so the verdict is a validated `{score, reasoning}` object, never hand-parsed text. See [Rubrics in depth](#rubrics-in-depth).
- **Regression tracking** — every run appends to `.eval_history.jsonl`; the next run diffs against the previous one and calls out, by test ID, any case that flipped from PASS to FAIL.
- **Rich terminal report** — a clean table with status, latency, token usage, and both check results per test case.
- **Zero billing risk** — runs on `gemini-2.5-flash` on the free tier. Once you hit the rate limit, calls fail with a 429; there is no path to an unexpected charge.
- **CI-friendly exit codes** — `judgedread` exits `0` when every case passes, `1` when any fails, and `2` on a usage error, so a prompt regression can fail a build.
- **Fully testable offline** — every test in this repo mocks the Gemini client. You can clone this repo and run the test suite with no API key at all.

## Quick start

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Get a free API key (no credit card required)
#    https://aistudio.google.com/apikey
export GEMINI_API_KEY="AIza..."

# 3. Run the example suite
judgedread examples/test_suite.json
```

You should see a Rich table with one row (`TC_001`), a pass-rate line, and a new `.eval_history.jsonl` file recording the run. The history file is written to the current directory; pass `--history PATH` to put it somewhere else.

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

`target_prompt_path` is resolved relative to the suite file, so you can run a suite from any directory. The prompt template file uses `${variable}` placeholders (Python's `string.Template` syntax), substituted from each test case's `input_variables` before the call is made. See `examples/system_prompt.txt` and `examples/test_suite.json` for a working reference.

`assertions` and `rubric` are both optional — a test case can run deterministic checks only, judge checks only, both, or (if you just want latency/token numbers) neither.

## Rubrics in depth

The deterministic checks answer *"is this output well-formed?"*. The rubric answers the harder question: *"is it actually right?"* This section covers what a rubric is in JudgeDread, how to write one that produces a useful signal, how to read the scores, and how to maintain rubrics as your prompt changes.

### What a rubric is (in this tool, specifically)

A rubric is a written definition of "good" for **one test case**, which a second model call (the *judge*) applies to the output. In the suite JSON it has three fields:

| Field | Purpose | Does the judge see it? |
|---|---|---|
| `metric` | A label for the quality dimension being graded, e.g. `technical_accuracy`, `tone`, `completeness`. | **No.** It's for humans reading the suite. |
| `criteria` | The actual grading instruction. | **Yes.** This is the only part of the rubric the judge reads. |
| `passing_threshold` | The minimum score (1–5) that counts as a pass. | No. It's applied in code after the verdict comes back. |

The judge gets exactly two things: your `criteria` and the raw output. Here is the full judge prompt (`judgedread/evaluators.py`):

```
You are an objective, strict QA Code Judge. Rate the following generated AI
output based on the provided Criteria. Provide a numerical score between 1
and 5, followed by a one-sentence reasoning explanation.

CRITERIA: {criteria}
AI OUTPUT TO EVALUATE: {output}
```

The judge returns a `JudgeVerdict` (`{score: int, reasoning: str}`) through Gemini's structured-output mode. The case passes the semantic check if `score >= passing_threshold`. The overall test case passes only if the deterministic assertions pass as well.

Two consequences follow from that design, and they shape everything below:

1. **The judge never sees the input.** It doesn't get your `input_variables`, your system prompt, or the `description`. If correctness depends on what the user asked for, your `criteria` has to restate it. "Does the endpoint handle the requested fields?" can't be graded. "Does the endpoint accept `email` and `phone`, both typed as strings with a format constraint?" can.
2. **The 1–5 scale has no anchors unless you give it some.** The judge prompt doesn't say what a 3 means versus a 4. Without anchors, the judge makes up its own scale, and that scale can drift between runs and between judge model versions.

### How to build a good rubric

**Make it checkable against the output alone.** Write the criteria as if you were handing the output to a contractor who has never seen the input. If they'd have to ask "what was the user asking for?", put that answer in the criteria.

**Ask about observable facts, not impressions.** "Is this a good API spec?" gives you noise. "Does the spec mark `email` as `format: email` and `phone` as a string with a pattern or format?" gives you signal. If you can't say what evidence in the output would earn a 5, the judge can't either.

**Anchor the scale.** This is the biggest improvement you can make. Spell out what the key score levels look like, especially the levels on either side of your threshold:

```json
"rubric": {
  "metric": "technical_accuracy",
  "criteria": "The user asked for a profile-creation endpoint that accepts email and phone. Score 5 if both fields are present, typed as strings, and each has a format or pattern constraint. Score 4 if both are present and typed as strings but only one has a constraint. Score 3 if both are present but neither has a constraint. Score 2 if either field is missing. Score 1 if the output is not an endpoint definition at all.",
  "passing_threshold": 4
}
```

Now a 4 means something specific. When the score moves, you know what changed in the output.

**Grade one dimension per rubric.** If your criteria say "accurate, concise, and polite", an output that's accurate but rude gets some blended score that tells you nothing about which part failed. Pick one dimension. Turning that into multiple rubrics is covered below.

**Don't ask the judge to check what code can check.** Valid JSON, required keys, banned words, and length limits all belong in `assertions`. Those checks are free, instant, and never wrong. Save the rubric for judgment calls: correctness, faithfulness to the request, reasoning quality, tone.

**Set the threshold on purpose.** Use `4` for "must be right", which is a sensible default for correctness. Use `3` for "acceptable but not ideal", which suits stylistic dimensions where you only want to catch real failures. A threshold of `5` is usually too strict: normal judge variance makes the test flaky. The exception is a rubric where 5 is a crisp, binary condition.

**Watch out for known judge biases.** LLM judges tend to reward length, confident tone, and nice formatting regardless of correctness (see [`docs/PRIMER.md`](docs/PRIMER.md) §3). If your rubric doesn't care about length, say so: "Do not reward length or formatting; grade only whether…".

### How to compare results against a rubric

A judge score is a noisy measurement. Treat it like one.

**Calibrate before you trust it.** When you write a new rubric, collect 10–20 real outputs, including some you know are bad, and score them yourself using the same criteria. Then run the judge on them and compare. If you and the judge disagree on pass/fail for more than one or two of them, the rubric is ambiguous. Fix the wording, not the threshold.

**Run it more than once.** Run the same suite 3–5 times without changing anything. A case that flips between PASS and FAIL across identical runs has a score sitting on the threshold, or a rubric loose enough to read either way. Either tighten the anchors or accept that the case is a weak signal. Don't chase a "regression" that's really just variance.

**Read the reasoning, not just the number.** The one-sentence `reasoning` is how you tell "the judge caught a real defect" apart from "the judge misread the rubric". Right now the CLI table shows only the score, and `.eval_history.jsonl` records only PASS/FAIL per test ID. To see the reasoning, call the harness from Python:

```python
from pathlib import Path

from google import genai
from judgedread.cli import build_results
from judgedread.engine import load_prompt_template, load_test_suite

suite_path = Path("examples/test_suite.json")
suite = load_test_suite(suite_path)
template = load_prompt_template(suite_path.parent / suite.target_prompt_path)
results = build_results(suite, template, genai.Client())
for r in results:
    if r.judge_verdict:
        print(r.id, r.judge_verdict.score, r.judge_verdict.reasoning)
```

**Compare by test ID across runs.** The regression check in `report.py` diffs the latest run against the previous one by test ID and reports any case that went from PASS to FAIL. A rubric failure on a case that used to pass, after a prompt edit, is the signal this tool exists to produce. But it's only meaningful if you've already confirmed (with the repeat runs above) that the case is stable when nothing changes.

### How to evolve a rubric over time

Rubrics are test code, and they need the same care as test code.

- **Version them with the prompt.** Keep the suite JSON in git next to the prompt template. When a rubric changes, put the reason in the commit message.
- **Don't change a rubric and the prompt in the same run.** If you do, a score change can't be attributed to either one. Change the rubric first, re-run against the *current* prompt to get a new baseline, and only then edit the prompt.
- **Tighten when it's too lenient.** If you find a bad output in the wild that the rubric passed, add an anchor that describes that failure ("Score 2 if the phone field is typed as an integer"). Then add a test case whose input reproduces it.
- **Loosen when it's too strict.** If the judge keeps failing outputs you'd ship, the rubric is encoding a requirement you don't actually have. Rewrite it, or drop the threshold for that dimension. Don't just retry until it passes.
- **Be careful about history after you edit a rubric.** History is keyed by test ID, so an edited rubric gets diffed against results produced by the old wording. After a meaningful rubric rewrite, expect a one-time flip, or give the case a new ID (`TC_001` → `TC_001_v2`) so the history makes the break explicit.
- **Re-calibrate when the judge model changes.** The judge is pinned to `JUDGE_MODEL = "gemini-2.5-flash"` in `evaluators.py`. A new judge model is a new measuring instrument, so re-run your hand-scored calibration set before you trust comparisons across that boundary.
- **Prune.** Delete a rubric that has never failed and duplicates a deterministic assertion. Rewrite one that fails randomly. Every judge call uses free-tier quota, so a rubric that doesn't change your decisions isn't worth running.

### When to use several rubrics (and when not to)

The schema allows **one rubric per test case**. That's deliberate: each judge call grades one dimension and returns one score, which is what keeps the verdict easy to interpret.

**Use several rubrics when** an output has independent quality dimensions that can fail on their own and that you'd act on differently. Examples: technical correctness *and* safety, or factual accuracy *and* customer-facing tone. A blended score hides which one regressed. Separate scores tell you.

**Don't use several when:**
- the second "dimension" is really a deterministic check in disguise (move it to `assertions`)
- the dimensions always move together (you're paying for two judge calls to learn one thing)
- you're adding rubrics to make a flaky case look more thorough (fix the flaky rubric first)

**How to do it today:** give each dimension its own test case with the same `input_variables`, a distinct ID, and a different rubric:

```json
{ "id": "TC_001_accuracy", "input_variables": { "user_query": "..." }, "rubric": { "metric": "technical_accuracy", "criteria": "...", "passing_threshold": 4 } },
{ "id": "TC_001_safety",   "input_variables": { "user_query": "..." }, "rubric": { "metric": "no_pii_leak",        "criteria": "...", "passing_threshold": 5 } }
```

Each of these cases generates its own output, so the two rubrics grade two different samples of the same prompt, not the same text. That costs an extra generation call, and the two scores can disagree partly because of sampling variance. For catching regressions this is usually fine, since you care whether *each* dimension holds up, but keep it in mind when comparing the scores. Native multi-rubric support (several rubrics graded against one shared output) is a reasonable future extension of `TestCase`.

Multiple *judges* are a different idea: several models score the same rubric and you average or vote. That reduces single-judge bias and is a known mitigation in the evals field ([`docs/PRIMER.md`](docs/PRIMER.md) §7). JudgeDread doesn't do it. For a single-developer regression suite, a well-anchored rubric and repeat runs get you most of the benefit for much less quota.

## Project structure

```
judgedread/
  models.py       # Pydantic schema: TestSuite, TestCase, Assertions, Rubric, results
  engine.py        # prompt hydration + Gemini invocation + timing/token capture
  evaluators.py     # deterministic checks + LLM-as-judge
  report.py       # Rich table rendering + history/regression tracking
  cli.py          # wiring: load suite -> run -> evaluate -> report
tests/            # all mock the Gemini client — no API key needed
examples/         # a working prompt + test suite you can run immediately
docs/             # primer, build notes, and original design docs
```

## Testing

```bash
pytest -v
```

The tests run with zero network access and zero API key — the CLI, engine, and evaluator tests all inject a mocked client. The only thing that touches the real API is the manual smoke-test step above. This also means CI needs no secrets: [`.github/workflows/tests.yml`](.github/workflows/tests.yml) lints with [ruff](https://docs.astral.sh/ruff/) and runs the full suite on every push and PR to `main`, on Python 3.10, 3.12, and 3.13.

To run the same lint checks locally:

```bash
ruff check . && ruff format --check .
```

## License

MIT — see [`LICENSE`](LICENSE).

## Further reading

- [`docs/PRIMER.md`](docs/PRIMER.md) — a from-the-ground-up explainer on why AI prompt testing is hard and how JudgeDread's approach (deterministic + LLM-as-judge) addresses it, aimed at anyone with general programming background but no AI/ML background assumed.
- [`docs/BUILD_NOTES.md`](docs/BUILD_NOTES.md) — how this project was actually built: process, key decisions, and known rough edges.
- [`docs/design/`](docs/design/) — the original design proposal and the implementation plan this project was built from.
