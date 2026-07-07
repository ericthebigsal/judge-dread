# Build Notes

How JudgeDread actually got built, the decisions behind it, and what's still rough. Written for whoever picks this repo up next — including future-me.

## Origin

JudgeDread started as a proposal (`proposal.md`) for an evals framework built on Google's Gemini SDK (`google-genai`, `gemini-2.5-flash`). The original spec called for a config-driven test suite, a deterministic assertion pipeline, an LLM-as-judge pipeline with structured output, and a Rich terminal reporter with regression tracking.

Along the way we detoured through an Anthropic/Claude-based version, then deliberately reverted to the original Gemini plan — **the free tier was the deciding factor.** A pay-as-you-go API key has no spend ceiling by default; a mis-scoped eval loop, a runaway retry, or just an unexpectedly large test suite can produce a real bill with no warning. Gemini's free tier caps you with a hard rate limit (HTTP 429) instead of a credit card, so the worst case is "the run fails," not "the run costs money." For a tool whose whole job is running lots of small LLM calls in a loop, that tradeoff was worth more than Claude's higher output quality.

## Process

The implementation followed a plan-then-execute workflow:

1. **Plan.** A full implementation plan was written up front (`docs/superpowers/plans/2026-07-06-judgedread-gemini.md`), breaking the work into 6 tasks with exact file paths, function signatures, and complete test code for each — so there was no ambiguity left for the implementation step to resolve.
2. **Subagent-driven execution.** Each task was implemented by a fresh subagent working from just its task brief (not the whole plan, not prior conversation), following TDD: write the failing test, confirm it fails for the right reason, implement, confirm it passes, commit.
3. **Per-task review.** After each task, a separate reviewer subagent checked the diff against the task's requirements (spec compliance) and against general code-quality standards, independent of the implementer's own self-assessment. All 6 tasks were approved on the first pass, with only Minor findings (see below).
4. **Whole-branch review.** After all 6 tasks landed, a final review pass — dispatched on a more capable model than the individual task reviews — looked at the *combined* diff, specifically checking things no single task's review could see: whether the modules actually compose correctly end to end.

That last step earned its keep. It found two **Important** issues that only exist at the seams between tasks (see below), both fixed in a single follow-up commit and re-verified before calling the work done.

## Key architectural decisions

- **Dependency injection for the API client, everywhere.** Every function that calls Gemini (`invoke_model`, `run_test_case`, `evaluate_judge`) takes the client as a parameter rather than constructing one internally. `judgedread/cli.py`'s `main()` is the *only* place that calls `genai.Client()`. This is what makes all 26 tests run with a mocked client and zero API key — nothing in the test suite needs real credentials or network access.
- **Structured output for the judge, not text parsing.** `evaluate_judge` uses Gemini's `response_schema` + `response.parsed` to get a validated `JudgeVerdict` object directly — there is no `json.loads()` on the judge's raw text anywhere. This eliminates an entire class of "the judge almost returned valid JSON" flakiness.
- **Per-test regression detection, not just an aggregate pass rate.** The original spec asked for exactly this ("highlights visual diffs when a prompt change causes a previously passing test to fail"), and it's worth calling out because it's easy to under-build: comparing only the aggregate pass rate between runs would miss a case where one test flips FAIL while another flips PASS and the rate stays flat. `judgedread/report.py` stores every test ID's status in `.eval_history.jsonl` and compares by ID against the immediately preceding run, naming any test that regressed specifically.
- **`string.Template`, not `str.format()`, for prompt hydration.** Prompt templates and judge outputs both routinely contain literal `{`/`}` (JSON examples, generated JSON output). `str.format()` would try to interpret those as format fields. `Template` with `${var}` syntax sidesteps the collision entirely.

## What the final review caught

The per-task reviews all passed, but the **whole-branch** review found that nothing in the `engine → evaluators → cli` chain tolerated an abnormal model response:

- If Gemini safety-blocks a response, `response.text` comes back `None`. `evaluate_deterministic(None, ...)` then calls `json.loads(None)`, which raises `TypeError` — not the `JSONDecodeError` the function catches — and that exception propagated all the way up, aborting the *entire suite run* and losing every already-computed result.
- If Gemini's structured decoding fails for the judge, `response.parsed` is `None`. The CLI layer did `judge_verdict.score` on that `None`, raising `AttributeError`, with the same blast radius.

Both were fixed with one mechanism: `build_results()` in `judgedread/cli.py` now wraps each test case's evaluation in a `try/except`, and a failure in one test case becomes a synthetic `FAIL` result (with the error message recorded) instead of crashing the whole run. This matters specifically *because* this is a regression harness — a tool whose job is "run reliably and tell you what changed" shouldn't itself be the thing that goes down when one API call misbehaves.

## Known rough edges (left as-is, on purpose)

These were flagged during review and deliberately not fixed — either because they're cosmetic, or because fixing them wasn't worth the scope creep for a first version:

- **Pytest collection warnings** on the Pydantic model names `TestCase`, `TestSuite`, `TestCaseResult` — pytest tries to collect them as test classes because of the `Test` prefix, and gives up harmlessly. Fixable with a `__test__ = False` class attribute or a `pyproject.toml` collection-config entry, if the noise ever bothers you.
- **`judge_cell` in the terminal report hardcodes a `/5` display** — fine as long as the judge always scores on a 1–5 scale (which the current rubric schema enforces via `passing_threshold` being compared against a fixed-range score), but would need a coordinated change if the scoring scale ever became configurable.
- **No retry/backoff on rate-limited calls.** Deliberate: adding automatic retries to a free-tier tool is exactly the kind of thing that could turn a rate-limit wall into a much longer, noisier failure. A 429 just fails the run today; you re-run when you're ready.
- **`main()` has no error handling** for a missing suite file or a network error — a bad path prints a raw Python traceback rather than a friendly CLI message. Acceptable for a local dev tool used by its own author; would be worth hardening if this ever gets other users.

## Testing philosophy

All 26 tests mock the Gemini client (`unittest.mock.Mock()` with hand-built fake responses) — there is no automated test that calls the real API. This was a deliberate tradeoff, confirmed during the final review: an automated live-API test would reintroduce exactly the cost/flakiness/rate-limit surface this project exists to avoid. The one place that touches the real API is a manual smoke-test step (`python -m judgedread.cli examples/test_suite.json` with a real `GEMINI_API_KEY` set), which is intentionally not part of `pytest`.
