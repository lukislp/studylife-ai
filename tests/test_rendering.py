from datetime import datetime

from studylife_ai.ingestion.rendering import (
    render_course,
    render_course_goal,
    render_note,
    render_session,
)
from studylife_ai.studylife.models import CourseDto, CourseGoalDto, StudyLifeNote, StudySessionDto


def _note(**overrides: object) -> StudyLifeNote:
    defaults: dict[str, object] = {
        "id": 1,
        "title": "Notiz",
        "content": "plain text",
        "created_at": datetime(2026, 8, 1, 10, 0),
        "updated_at": datetime(2026, 8, 1, 10, 0),
        "course_id": None,
        "session_id": None,
        "is_markdown": False,
    }
    defaults.update(overrides)
    return StudyLifeNote(**defaults)  # type: ignore[arg-type]


def _course(**overrides: object) -> CourseDto:
    defaults: dict[str, object] = {
        "id": 6,
        "semester": 3,
        "name": "Lineare Algebra",
        "code": "MATH101",
        "topics": [],
        "ects": 5,
        "group": None,
    }
    defaults.update(overrides)
    return CourseDto(**defaults)  # type: ignore[arg-type]


def _session(**overrides: object) -> StudySessionDto:
    defaults: dict[str, object] = {
        "id": 42,
        "course_id": 6,
        "course_name": "Lineare Algebra",
        "start_time": datetime(2026, 8, 1, 10, 0),
        "end_time": datetime(2026, 8, 1, 11, 30),
        "topic": None,
        "notes": None,
        "is_completed": False,
    }
    defaults.update(overrides)
    return StudySessionDto(**defaults)  # type: ignore[arg-type]


def _course_goal(**overrides: object) -> CourseGoalDto:
    defaults: dict[str, object] = {
        "course_id": 6,
        "course_name": "Lineare Algebra",
        "target_date": None,
        "completion_note": None,
        "completed_at": None,
        "grade": None,
        "completed_topics": "",
        "tag": None,
    }
    defaults.update(overrides)
    return CourseGoalDto(**defaults)  # type: ignore[arg-type]


def test_render_course_without_group_or_topics() -> None:
    text = render_course(_course())

    assert text == "Course: Lineare Algebra (MATH101)\nSemester 3, 5 ECTS"


def test_render_course_with_group_and_topics() -> None:
    text = render_course(_course(group="Wahlpflicht Mathe", topics=["Eigenwerte", "Matrizen"]))

    assert text == (
        "Course: Lineare Algebra (MATH101)\n"
        "Semester 3, 5 ECTS, Elective group: Wahlpflicht Mathe\n"
        "Topics: Eigenwerte, Matrizen"
    )


def test_render_session_planned_without_topic_or_notes() -> None:
    text = render_session(_session())

    assert text == (
        "Study session: Lineare Algebra, 2026-08-01 10:00 - 2026-08-01 11:30\nStatus: Planned"
    )


def test_render_session_completed_with_topic_and_notes() -> None:
    text = render_session(_session(is_completed=True, topic="Eigenwerte", notes="Alles verstanden"))

    assert text == (
        "Study session: Lineare Algebra, 2026-08-01 10:00 - 2026-08-01 11:30\n"
        "Status: Completed\n"
        "Topic: Eigenwerte\n"
        "Notes: Alles verstanden"
    )


def test_render_course_goal_with_no_optional_fields() -> None:
    text = render_course_goal(_course_goal())

    assert text == "Course goal: Lineare Algebra"


def test_render_note_non_markdown_passes_content_through_unchanged() -> None:
    text = render_note(_note(content="# not actually markdown\n**stays raw**", is_markdown=False))

    assert text == "# not actually markdown\n**stays raw**"


def test_render_note_markdown_strips_syntax() -> None:
    content = "# Heading\n\nSome **bold** and *italic* text with `inline code`.\n"
    text = render_note(_note(content=content, is_markdown=True))

    assert text == "Heading\n\nSome bold and italic text with inline code."


def test_render_note_markdown_list_items_get_separated() -> None:
    text = render_note(_note(content="- item one\n- item two\n", is_markdown=True))

    assert "item one" in text
    assert "item two" in text
    assert "item oneitem two" not in text


def test_render_note_markdown_link_keeps_only_the_text() -> None:
    text = render_note(_note(content="[a link](https://example.com)", is_markdown=True))

    assert text == "a link"
    assert "https://example.com" not in text


def test_render_course_goal_with_all_optional_fields() -> None:
    text = render_course_goal(
        _course_goal(
            target_date=datetime(2026, 9, 1),
            grade=1.3,
            completed_at=datetime(2026, 9, 2, 14, 0),
            completed_topics="Eigenwerte, Matrizen",
            completion_note="Nochmal Determinanten üben",
            tag="wichtig",
        )
    )

    assert text == (
        "Course goal: Lineare Algebra\n"
        "Target date: 2026-09-01\n"
        "Grade: 1.3\n"
        "Completed at: 2026-09-02 14:00\n"
        "Completed topics: Eigenwerte, Matrizen\n"
        "Note: Nochmal Determinanten üben\n"
        "Tag: wichtig"
    )
