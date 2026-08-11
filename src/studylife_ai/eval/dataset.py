"""Loads the versioned RAGAS eval set from eval/dataset.jsonl."""

from pathlib import Path

from pydantic import BaseModel

DEFAULT_DATASET_PATH = Path("eval/dataset.jsonl")


class EvalCase(BaseModel):
    """One eval question, with the entity titles retrieval should surface (empty = none
    expected). Named `expected_titles`, not `expected_note_titles` - cases can expect
    course/session/course_goal titles too, not just notes (see docs/decisions.md "Eval coverage
    for course/session questions")."""

    id: str
    question: str
    expected_titles: list[str]
    category: str


def load_eval_cases(path: Path = DEFAULT_DATASET_PATH) -> list[EvalCase]:
    with path.open(encoding="utf-8") as f:
        return [EvalCase.model_validate_json(line) for line in f if line.strip()]
