"""A stand-in for the google-genai client that runs models through a local Ollama server.

Only the slice of the client that JudgeDread uses is implemented:
``client.models.generate_content(model=..., contents=..., config=...)``.
"""

import json
from types import SimpleNamespace
from urllib.error import URLError
from urllib.request import Request, urlopen

from pydantic import ValidationError

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_LOCAL_MODEL = "gemma3:4b"


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, host: str = DEFAULT_HOST):
        self.host = host.rstrip("/")
        self.models = _Models(self)

    def _post(self, path: str, payload: dict) -> dict:
        request = Request(
            f"{self.host}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request) as response:
            return json.loads(response.read())

    def check_model(self, model: str) -> None:
        """Raise OllamaError unless the server is up and `model` has been pulled."""
        try:
            with urlopen(f"{self.host}/api/tags") as response:
                tags = json.loads(response.read())
        except URLError as exc:
            raise OllamaError(
                f"Ollama is not reachable at {self.host} ({exc.reason}). Start the Ollama app."
            ) from exc

        # A bare name like "mistral" means the ":latest" tag.
        wanted = model if ":" in model else f"{model}:latest"
        if wanted not in {m["name"] for m in tags.get("models", [])}:
            raise OllamaError(f"model {model!r} is not downloaded. Run: ollama pull {model}")


class _Models:
    def __init__(self, client: OllamaClient):
        self._client = client

    def generate_content(self, model: str, contents: str, config) -> SimpleNamespace:
        payload = {"model": model, "prompt": contents, "stream": False}
        if config.max_output_tokens:
            payload["options"] = {"num_predict": config.max_output_tokens}
        schema = config.response_schema
        if schema is not None:
            payload["format"] = schema.model_json_schema()

        data = self._client._post("/api/generate", payload)
        text = data["response"]

        parsed = None
        if schema is not None:
            try:
                parsed = schema.model_validate_json(text)
            except ValidationError:
                pass

        return SimpleNamespace(
            text=text,
            parsed=parsed,
            usage_metadata=SimpleNamespace(
                prompt_token_count=data.get("prompt_eval_count", 0),
                candidates_token_count=data.get("eval_count", 0),
            ),
        )
