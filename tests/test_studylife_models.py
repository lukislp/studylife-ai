from studylife_ai.studylife.models import CourseDto, CourseGoalDto, StudyLifeNote, StudySessionDto


def test_study_life_note_parses_camel_case_and_defaults_is_markdown_to_false() -> None:
    note = StudyLifeNote.model_validate(
        {
            "id": 1,
            "title": "Notiz",
            "content": "text",
            "createdAt": "2026-08-01T10:00:00",
            "updatedAt": "2026-08-01T10:00:00",
        }
    )

    assert note.course_id is None
    assert note.session_id is None
    assert note.is_markdown is False
    assert note.source_url is None


def test_study_life_note_parses_is_markdown_true() -> None:
    note = StudyLifeNote.model_validate(
        {
            "id": 1,
            "title": "Notiz",
            "content": "# text",
            "createdAt": "2026-08-01T10:00:00",
            "updatedAt": "2026-08-01T10:00:00",
            "isMarkdown": True,
        }
    )

    assert note.is_markdown is True


def test_study_life_note_parses_source_url() -> None:
    note = StudyLifeNote.model_validate(
        {
            "id": 1,
            "title": "Notiz",
            "content": "text",
            "createdAt": "2026-08-01T10:00:00",
            "updatedAt": "2026-08-01T10:00:00",
            "sourceUrl": "https://example.com/article",
        }
    )

    assert note.source_url == "https://example.com/article"


def test_course_dto_parses_camel_case_and_applies_defaults() -> None:
    course = CourseDto.model_validate(
        {"id": 6, "semester": 3, "name": "Lineare Algebra", "code": "MATH101"}
    )

    assert course.id == 6
    assert course.topics == []
    assert course.ects == 5
    assert course.group is None


def test_course_dto_parses_all_fields() -> None:
    course = CourseDto.model_validate(
        {
            "id": 6,
            "semester": 3,
            "name": "Lineare Algebra",
            "code": "MATH101",
            "topics": ["Eigenwerte", "Matrizen"],
            "ects": 5,
            "group": "Wahlpflicht Mathe",
        }
    )

    assert course.topics == ["Eigenwerte", "Matrizen"]
    assert course.group == "Wahlpflicht Mathe"


def test_study_session_dto_parses_camel_case_and_applies_defaults() -> None:
    session = StudySessionDto.model_validate(
        {
            "id": 42,
            "courseId": 6,
            "startTime": "2026-08-01T10:00:00",
            "endTime": "2026-08-01T11:30:00",
        }
    )

    assert session.course_id == 6
    assert session.course_name == ""
    assert session.topic is None
    assert session.notes is None
    assert session.is_completed is False


def test_study_session_dto_parses_all_fields() -> None:
    session = StudySessionDto.model_validate(
        {
            "id": 42,
            "courseId": 6,
            "courseName": "Lineare Algebra",
            "startTime": "2026-08-01T10:00:00",
            "endTime": "2026-08-01T11:30:00",
            "topic": "Eigenwerte",
            "notes": "Alles verstanden",
            "isCompleted": True,
        }
    )

    assert session.topic == "Eigenwerte"
    assert session.notes == "Alles verstanden"
    assert session.is_completed is True


def test_course_goal_dto_parses_camel_case_with_all_optional_fields_absent() -> None:
    goal = CourseGoalDto.model_validate({"courseId": 6})

    assert goal.course_id == 6
    assert goal.course_name == ""
    assert goal.target_date is None
    assert goal.completion_note is None
    assert goal.completed_at is None
    assert goal.grade is None
    assert goal.completed_topics == ""
    assert goal.tag is None


def test_course_goal_dto_parses_all_fields() -> None:
    goal = CourseGoalDto.model_validate(
        {
            "courseId": 6,
            "courseName": "Lineare Algebra",
            "targetDate": "2026-09-01T00:00:00",
            "completionNote": "Nochmal Determinanten üben",
            "completedAt": "2026-09-02T14:00:00",
            "grade": 1.3,
            "completedTopics": "Eigenwerte, Matrizen",
            "tag": "wichtig",
        }
    )

    assert goal.target_date is not None
    assert goal.grade == 1.3
    assert goal.completed_topics == "Eigenwerte, Matrizen"
    assert goal.tag == "wichtig"
