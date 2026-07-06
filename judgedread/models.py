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
