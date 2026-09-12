"""Atheris fuzz harness for the prompt-injection escaping and the request models.

Contract under test: escape_untrusted_text() never raises, never leaves a literal `<`/`>`
behind (the boundary characters the DATA framing relies on) and is idempotent; the request
models either validate a JSON body or raise pydantic's ValidationError (which FastAPI turns
into a 422) - never anything else.

Run locally (Linux, needs the atheris wheel):
    uv sync --frozen --group fuzz
    uv run python fuzz/fuzz_prompt_and_schemas.py -max_total_time=60
CI runs the same harness for a short, fixed time budget (see .github/workflows/ci.yml).
"""

from __future__ import annotations

import contextlib
import sys

import atheris
from pydantic import ValidationError

from studylife_ai.schemas.agent import AgentRequest, ConfirmRequest
from studylife_ai.schemas.chat import ChatRequest
from studylife_ai.schemas.internal import EnrichCaptureRequest, RegisterKeyRequest
from studylife_ai.text_escaping import escape_untrusted_text

MODELS = (AgentRequest, ConfirmRequest, ChatRequest, EnrichCaptureRequest, RegisterKeyRequest)


def test_one_input(data: bytes) -> None:
    fdp = atheris.FuzzedDataProvider(data)
    text = fdp.ConsumeUnicodeNoSurrogates(256)
    escaped = escape_untrusted_text(text)
    if "<" in escaped or ">" in escaped or escape_untrusted_text(escaped) != escaped:
        raise AssertionError(f"escape_untrusted_text left a boundary character in {escaped!r}")
    body = fdp.ConsumeBytes(512)
    for model in MODELS:
        with contextlib.suppress(ValidationError):
            model.model_validate_json(body)


if __name__ == "__main__":
    # instrument_all() instead of instrument_imports(): the package is loaded through uv's
    # editable-install loader, which the import hook does not see (no coverage feedback,
    # so libFuzzer would never grow its inputs past a few bytes).
    atheris.instrument_all()
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()
