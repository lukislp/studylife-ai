"""Loads the versioned RAGAS eval set from eval/dataset.jsonl."""

from pathlib import Path

from pydantic import BaseModel

DEFAULT_DATASET_PATH = Path("eval/dataset.jsonl")


class EvalCase(BaseModel):
    """One eval question, with the notes it should retrieve (empty = none expected)."""

    id: str
    question: str
    expected_note_titles: list[str]
    category: str


def load_eval_cases(path: Path = DEFAULT_DATASET_PATH) -> list[EvalCase]:
    with path.open(encoding="utf-8") as f:
        return [EvalCase.model_validate_json(line) for line in f if line.strip()]
