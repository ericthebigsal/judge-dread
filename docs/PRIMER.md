# A Primer on AI Prompt Evaluation

*Pitched at roughly a 200-level college course: you know how to program and you know what a unit test is, but you haven't necessarily worked with large language models before. This document explains the ideas behind JudgeDread from first principles, not just how to use it.*

## 1. Why you can't just `assert output == expected`

In ordinary software, a function is deterministic: given the same input, `add(2, 3)` returns `5` every single time, forever. That's what makes unit testing work — you write down the expected output once, and an equality check is a permanent, reliable oracle.

A large language model (LLM) is not that kind of function. Ask GPT-, Claude-, or Gemini-family models the same question twice and you will usually get two *different* strings back, even with identical input. This isn't a bug — it's how these models are designed to work.

Here's the mechanism, briefly: an LLM doesn't compute "the answer." At each step, it computes a **probability distribution over the next token** (roughly, the next word-piece) given everything so far, then samples from that distribution. A parameter called **temperature** controls how "peaky" that sampling is — temperature 0 pushes the model toward always picking the single most likely token (nearly deterministic, though not always perfectly so across hardware/batching), while higher temperatures introduce real randomness, favoring variety over predictability. Most production use sits somewhere in between, because a small amount of variety is often desirable — it's what makes chat responses feel natural instead of robotic.

The consequence for testing: you cannot pin down an LLM's output the way you pin down `add(2, 3)`. Two runs of the same prompt might both be *good* answers while being *completely different strings*. `assert output == expected_string` is simply the wrong tool.

## 2. Two different questions you can ask about an output

Given that exact-match doesn't work, what *can* you check? It turns out there are two genuinely different kinds of questions you can ask about a piece of AI output, and they call for different tools.

### 2a. Structural / mechanical questions

*"Is this output well-formed?"* Examples:
- Is it valid JSON?
- Does it contain the required fields?
- Does it avoid a list of banned words or phrases?
- Is it under some length limit?

These questions have a definite, computable, deterministic answer — you don't need another AI model to check them, and you shouldn't use one, because a hand-written check is faster, free, and never itself hallucinates a wrong verdict. JudgeDread's `evaluate_deterministic()` (in `judgedread/evaluators.py`) is exactly this: `json.loads()` for validity, a membership check for required keys, a regex search for banned phrases. Nothing fancy, and that's the point — deterministic checks should be the boring, bulletproof layer.

### 2b. Semantic / quality questions

*"Is this output actually good?"* Examples:
- Does this generated API spec correctly require an email field to be a string?
- Is this summary faithful to the source document?
- Is this customer-support reply appropriately empathetic?

These questions don't reduce to a regex. "Correctly requires a string format" requires understanding what the text *means*, not just its shape. This is where the second technique comes in.

## 3. LLM-as-judge

The core idea: use a second LLM call to *evaluate* the first LLM's output, by giving the judge model a **rubric** (a written description of what "good" means for this specific case) and asking it to score the output against that rubric.

```
You are an objective, strict QA Code Judge. Rate the following generated AI
output based on the provided Criteria. Provide a numerical score between 1
and 5, followed by a one-sentence reasoning explanation.

CRITERIA: {criteria}
AI OUTPUT TO EVALUATE: {output}
```

This works surprisingly well in practice — modern LLMs are reasonably good at following an explicit grading rubric, and it scales in a way that human review doesn't (you can run thousands of judged evaluations for the cost of an API call each, instead of paying a human to read every output). It's become a standard technique in the AI evals field, not a JudgeDread invention.

It also has real, well-documented limitations, and a 200-level treatment shouldn't gloss over them:

- **The judge is also a non-deterministic LLM.** It can be wrong, inconsistent between runs, or miscalibrated (too lenient, too harsh, or harsh on some categories and lenient on others).
- **Judges have measurable biases.** Published research has found LLM judges systematically favor longer answers, favor answers stylistically similar to their own outputs, and can be swayed by superficial features (confident tone, formatting) independent of actual correctness.
- **Self-preference bias.** A model asked to judge outputs — including its own family's outputs — can rate them more favorably than an independent judge would, which is part of why it's common practice to use a *different, often more capable* model as the judge than the one being evaluated.
- **A rubric is only as good as its author.** "Is this a good summary?" is nearly useless as a rubric — it's not gradeable. "Does the summary mention the deal's dollar amount and closing date?" is. Writing gradeable, specific rubrics is a skill, and vague rubrics produce noisy, low-value scores no matter how good the judge model is.

None of this means LLM-as-judge is unusable — it means you should treat a judge score the way you'd treat any noisy measurement: useful in aggregate, over many samples, with a rubric you trust, and not as a single infallible ground truth for one run.

## 4. Structured output: making the judge's answer machine-readable

There's a subtle engineering problem hiding in the judge pattern above: the judge's response is *itself* just more free-form LLM text. If you ask a model to "give a score and a reasoning," you might get back something like:

```
I would rate this a 4 out of 5, since the email field correctly...
```

Now you're back to square one — you need to parse natural language to extract a number, and that parsing step can itself fail (what if it says "four" instead of "4"? What if there's no clean sentence boundary?).

**Structured output** (sometimes called "constrained decoding" or "JSON mode") solves this by having the API itself force the model's output to conform to a schema you specify — in JudgeDread's case, a Pydantic model:

```python
class JudgeVerdict(BaseModel):
    score: int
    reasoning: str
```

The Gemini API is told `response_schema=JudgeVerdict`, and the returned `response.parsed` is already a validated `JudgeVerdict` instance — no regex, no `json.loads()`, no "the model almost returned valid JSON but added a trailing comma" failure mode. This is a meaningfully different (and more reliable) technique than asking a model to "please respond in JSON" and hoping — the schema is *enforced* by the generation process itself, not just requested by the prompt.

## 5. Regression testing, translated to prompts

If you've written traditional software tests, you already understand the core idea here: a regression is when something that used to work stops working. The value of a test *suite* — as opposed to one-off manual checks — is that it turns "did I break something?" from a question you have to guess at into a question you can compute.

The same idea applies to prompts, with one added wrinkle: because outputs are non-deterministic, "did this test pass" is itself a slightly fuzzier judgment (a deterministic check is exact; a judge score crossing a threshold is a probabilistic-ish signal). But the *structure* of regression testing carries over cleanly:

1. Run the suite, get a PASS/FAIL per test case.
2. Record the result.
3. Next time you change the prompt, run the suite again.
4. Compare: did anything that passed before, fail now?

The naive version of step 4 compares only the *aggregate* pass rate (e.g., "80% passed last time, 75% pass now — regression!"). That's a real signal, but it's a lossy one: imagine test A flips from PASS to FAIL while test B flips from FAIL to PASS in the same run. The aggregate pass rate can stay *exactly the same* while a real, specific regression happened. JudgeDread's `report.py` stores every test ID's status per run in `.eval_history.jsonl` and diffs by ID, so it can tell you *which specific test* regressed — the same way a CI system tells you which specific test failed, not just "3 fewer tests passed than yesterday."

## 6. Putting it together

JudgeDread's pipeline, end to end, is now hopefully legible as an application of all five ideas above:

```
test_suite.json  →  hydrate prompt template (§1: templating around non-determinism)
                  →  call Gemini            (§1: the non-deterministic step)
                  →  deterministic checks   (§2a: cheap, exact, structural)
                  →  LLM-as-judge checks    (§2b + §3: expensive, fuzzy, semantic)
                       ↳ via structured output (§4: no text-parsing fragility)
                  →  Rich report + history  (§5: regression tracking by test ID)
```

Nothing here is exotic — the whole project is a composition of five well-understood ideas, wired together in about 250 lines of Python across five files. That's a deliberate design choice, not an accident: a small, legible pipeline that you can read end-to-end in ten minutes is more trustworthy than a framework that hides the same five ideas behind abstraction.

## 7. Open questions, if you want to go further

This primer stops at "how JudgeDread works." If you want to go deeper into the field this sits in (often called "AI evals" or "LLM evaluation"), some genuinely open, actively-researched questions worth knowing exist:

- **Judge calibration** — how do you know if your judge's "4/5" actually corresponds to the quality you'd assign it yourself? Techniques like periodically spot-checking judge scores against human ratings, or using an ensemble of multiple judge models, are common mitigations, not solved problems.
- **Benchmark/rubric contamination** — if a rubric or test suite becomes well-known, later model versions can end up (intentionally or not) trained on data that makes them better at *that specific test* without being better in general. This is a bigger concern for public benchmarks than private in-house eval suites like this one, but the underlying risk (optimizing for the test instead of the goal) is the same failure mode Goodhart's Law describes in any measurement system.
- **Statistical significance with small samples** — a single test case passing or failing tells you much less than a traditional unit test would, because of the non-determinism discussed in §1. Running each case multiple times and looking at pass *rate* rather than a single pass/fail is a natural extension this project doesn't currently implement.
- **Cost/latency evaluation as a first-class citizen** — JudgeDread already tracks latency and token counts per test case; a natural next step for a more mature harness is treating "did this get slower or more expensive" as its own kind of regression, not just a side metric.

None of these need to block you from using or extending a tool like this one — they're the kind of thing worth knowing exists as you build more sophisticated eval infrastructure later.
