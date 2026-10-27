# Build Notes

How JudgeDread actually got built, the decisions behind it, and what's still rough. Written for whoever picks this repo up next — including future-me.

## Origin

JudgeDread started as a proposal ([`design/proposal.md`](design/proposal.md)) for an evals framework built on Google's Gemini SDK (`google-genai`, `gemini-2.5-flash`). The original spec called for a config-driven test suite, a deterministic assertion pipeline, an LLM-as-judge pipeline with structured output, and a Rich terminal reporter with regression tracking.

Along the way we detoured through an Anthropic/Claude-based version, then deliberately reverted to the original Gemini plan — **the free tier was the deciding factor.** A pay-as-you-go API key has no spend ceiling by default; a mis-scoped eval loop, a runaway retry, or just an unexpectedly large test suite can produce a real bill with no warning. Gemini's free tier caps you with a hard rate limit (HTTP 429) instead of a credit card, so the worst case is "the run fails," not "the run costs money." For a tool whose whole job is running lots of small LLM calls in a loop, that tradeoff was worth more than Claude's higher output quality.

## Process

The implementation followed a plan-then-execute workflow:

1. **Plan.** A full implementation plan was written up front ([`design/2026-07-06-implementation-plan.md`](design/2026-07-06-implementation-plan.md)), breaking the work into 6 tasks with exact file paths, function signatures, and complete test code for each — so there was no ambiguity left for the implementation step to resolve.
2. **Subagent-driven execution.** Each task was implemented by a fresh subagent working from just its task brief (not the whole plan, not prior conversation), following TDD: write the failing test, confirm it fails for the right reason, implement, confirm it passes, commit.
3. **Per-task review.** After each task, a separate reviewer subagent checked the diff against the task's requirements (spec compliance) and against general code-quality standards, independent of the implementer's own self-assessment. All 6 tasks were approved on the first pass, with only Minor findings (see below).
4. **Whole-branch review.** After all 6 tasks landed, a final review pass — dispatched on a more capable model than the individual task reviews — looked at the *combined* diff, specifically checking things no single task's review could see: whether the modules actually compose correctly end to end.

That last step earned its keep. It found two **Important** issues that only exist at the seams between tasks (see below), both fixed in a single follow-up commit and re-verified before calling the work done.

## Key architectural decisions

- **Dependency injection for the API client, everywhere.** Every function that calls Gemini (`invoke_model`, `run_test_case`, `evaluate_judge`) takes the client as a parameter rather than constructing one internally. `judgedread/cli.py`'s `main()` is the *only* place that calls `genai.Client()`. This is what makes every test run with a mocked client and zero API key — nothing in the test suite needs real credentials or network access.
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

- **`judge_cell` in the terminal report hardcodes a `/5` display** — fine as long as the judge always scores on a 1–5 scale (which the current rubric schema enforces via `passing_threshold` being compared against a fixed-range score), but would need a coordinated change if the scoring scale ever became configurable.
- **No retry/backoff on rate-limited calls.** Deliberate: adding automatic retries to a free-tier tool is exactly the kind of thing that could turn a rate-limit wall into a much longer, noisier failure. A 429 just fails the run today; you re-run when you're ready.

## Post-v0.1 cleanup

A later cleanup pass resolved two of the rough edges originally listed above:

- The Pydantic models named `Test*` set `__test__ = False`, so pytest no longer warns about trying to collect them.
- `main()` is now the `judgedread` console script. It reports a missing or invalid suite file and a missing `GEMINI_API_KEY` on stderr rather than as a traceback, and it returns a meaningful exit code (`0` all passed, `1` any failed, `2` usage error) so the harness can gate CI.

The same pass made `target_prompt_path` resolve relative to the suite file instead of the current directory, and added ruff linting to CI.

## Local mode (Ollama)

The free tier that made Gemini the right default also became the bottleneck: at 20 requests a day, and two calls per judged case, smoke tests kept running out of quota. `--local` (PR #4) runs **both** calls, the prompt and the judge, on [Ollama](https://ollama.com) on the developer's machine, so a smoke test needs no Gemini at all. Gemini stays the default; local mode is opt-in.

- **A stand-in client, not a backend abstraction.** `judgedread/ollama.py`'s `OllamaClient` implements just the slice of the google-genai client JudgeDread touches: `client.models.generate_content(model=..., contents=..., config=...)`, returning an object with `.text`, `.parsed`, and `.usage_metadata`. Because every call site already took the client as a parameter (see *Dependency injection* above), `engine.py` and `evaluators.py` didn't change at all. The only wiring is in `cli.py`, which picks the client and threads `model` / `judge_model` through `build_results()`. It uses stdlib `urllib`, so there's no new dependency.
- **Structured output survives.** The judge's `response_schema` (the `JudgeVerdict` Pydantic model) is passed to Ollama's `format` field as a JSON schema, and the reply is validated back into a `JudgeVerdict`. A reply that doesn't validate becomes `parsed = None`, which hits the existing "judge did not return a parsable verdict" path and becomes a per-case `FAIL`, the same as on Gemini.
- **Fail fast before the run.** `OllamaClient.check_model()` hits `/api/tags` before any test case runs. If Ollama isn't running or a model hasn't been pulled, the CLI exits `2` with the exact `ollama pull <model>` command. This is better than letting every case fail with a connection error.

### Why two local models

The target machine is an M3 with 8 GB of memory, which limits you to ~3–4B-parameter models. The plan was a single default model. Measurement killed that plan: across repeated runs of both example suites, neither small model did both jobs well.

| | Runs the prompt | Judges |
|---|---|---|
| `gemma3:4b` | wrapped JSON in markdown fences, 6/6 runs | consistent 5/5, sound reasoning |
| `qwen2.5:3b` | bare JSON, 6/6 runs | scored sound outputs 3/5 in 2 of 5 runs, with reasoning that was simply wrong |

The qwen judge failures were checked by pairing each output with its score. The low scores didn't track weaker outputs. For example, it gave a correct E.164 phone pattern a 3 and said the field "should be replaced with a 'string' type" when it already was one. So the defaults are split: `qwen2.5:3b` runs the prompt and `gemma3:4b` judges, each overridable with `--local-model` / `--local-judge-model`. With the split, the judge scored 5/5 on every run. Ollama may swap the two models in and out of memory mid-run, costing a few seconds. That's acceptable for a smoke test. `qwen3:4b` was ruled out up front because it emits thinking text by default, which would land in the output the assertions check.

### What was deliberately not done

- **No fence-stripping or JSON repair in the harness.** `qwen2.5:3b` still fails the JSON check in roughly 1 run in 4, because it writes regex backslashes unescaped inside JSON strings (`"^\d{10,15}$"`). That's genuinely invalid JSON, and so is gemma's fenced output. Anything downstream that calls `json.loads` on this prompt's output would break the same way, and loosening the check would hide that failure on Gemini too. The example prompt wasn't changed to suit small models either, because the prompt is the thing under test.
- **No mixing Gemini with local** (e.g. Gemini prompt + local judge), no automatic model downloads, no other runners (LM Studio etc.), and no retries.
- **Local scores aren't comparable to Gemini's.** A 4B judge is a different and more lenient measuring instrument. Local mode is for "does the suite run and is anything obviously broken," not a replacement for Gemini verdicts. Run history isn't separated by backend, so don't read a PASS→FAIL flip across a backend switch as a regression.

## Testing philosophy

All tests mock the Gemini client (`unittest.mock.Mock()` with hand-built fake responses) — there is no automated test that calls the real API. This was a deliberate tradeoff, confirmed during the final review: an automated live-API test would reintroduce exactly the cost/flakiness/rate-limit surface this project exists to avoid. The places that touch a real model are manual smoke-test steps, intentionally not part of `pytest`: `judgedread examples/test_suite.json` with a real `GEMINI_API_KEY` set, or `judgedread examples/test_suite.json --local` against Ollama. The Ollama tests follow the same rule and patch `urlopen`, so CI needs neither an API key nor a running Ollama.
