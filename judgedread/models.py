from pydantic import BaseModel, Field


class Assertions(BaseModel):
    expected_format: str | None = None
    required_keys: list[str] = Field(default_factory=list)
    banned_phrases: list[str] = Field(default_factory=list)


class Rubric(BaseModel):
    metric: str
    criteria: str
    passing_threshold: int


class TestCase(BaseModel):
    __test__ = False  # not a pytest test class, despite the name

    id: str
    description: str
    input_variables: dict[str, str] = Field(default_factory=dict)
    assertions: Assertions = Field(default_factory=Assertions)
    rubric: Rubric | None = None


class TestSuite(BaseModel):
    __test__ = False  # not a pytest test class, despite the name

    suite_name: str
    target_prompt_path: str
    test_cases: list[TestCase]


class DeterministicResult(BaseModel):
    passed: bool
    failures: list[str] = Field(default_factory=list)
    parsed_json: dict | None = None


class JudgeVerdict(BaseModel):
    score: int
    reasoning: str


class TestCaseResult(BaseModel):
    __test__ = False  # not a pytest test class, despite the name

    id: str
    description: str
    status: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    deterministic: DeterministicResult
    judge_verdict: JudgeVerdict | None = None
    raw_output: str
