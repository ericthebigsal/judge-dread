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
