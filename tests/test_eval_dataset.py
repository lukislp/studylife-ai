from pathlib import Path

from studylife_ai.eval.dataset import EvalCase, load_eval_cases


def test_load_eval_cases_parses_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    path.write_text(
        '{"id": "a", "question": "Was sind Eigenwerte?", '
        '"expected_titles": ["Merkzettel Eigenwerte"], "category": "basic"}\n'
        "\n"
        '{"id": "b", "question": "Kein Treffer?", "expected_titles": [], '
        '"category": "no-match"}\n',
        encoding="utf-8",
    )

    cases = load_eval_cases(path)

    assert cases == [
        EvalCase(
            id="a",
            question="Was sind Eigenwerte?",
            expected_titles=["Merkzettel Eigenwerte"],
            category="basic",
        ),
        EvalCase(id="b", question="Kein Treffer?", expected_titles=[], category="no-match"),
    ]
