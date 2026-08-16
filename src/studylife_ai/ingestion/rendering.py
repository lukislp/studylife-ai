"""Renders StudyLife courses/sessions/course goals into embeddable text.

Unlike notes (freeform text a user typed), these are structured records —
each render function turns one entity into a single natural-language block
in the same shape a chunk of note content would take, so the rest of the
pipeline (chunking, embedding, prompt building) doesn't need to know the
difference. Templates approved by the user (see docs/decisions.md):
English labels for consistency with the rest of the codebase, since the
underlying data (course names, topics, notes) is whatever language the
user wrote it in - same language-neutral approach as the RAG prompt itself.
"""

import re
from html.parser import HTMLParser

import mistune

from studylife_ai.studylife.models import CourseDto, CourseGoalDto, StudyLifeNote, StudySessionDto

_DATE_FORMAT = "%Y-%m-%d"
# Public - sync.py's session title also needs this, to stay in sync with
# render_session()'s own use of it rather than duplicating the literal.
DATETIME_FORMAT = "%Y-%m-%d %H:%M"

# "table" matches Markdig's default pipe-table support (StudyLife's Markdown renderer) - without
# it mistune would leave table syntax as literal text instead of parsing it into <table> markup.
_markdown_to_html = mistune.create_markdown(plugins=["table"])

# Block-level tags get a leading newline so e.g. two list items or a heading followed by a
# paragraph don't run together into one sentence once tags are stripped.
_BLOCK_TAGS = frozenset(
    {"p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "tr", "pre", "br"}
)
_BLANK_LINES = re.compile(r"\n{3,}")


class _HtmlTextExtractor(HTMLParser):
    """Strips tags from mistune's HTML output, keeping block structure as newlines."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        return _BLANK_LINES.sub("\n\n", "".join(self._parts)).strip()


def render_note(note: StudyLifeNote) -> str:
    """Notes are freeform text a user typed - Markdown-mode ones (see docs/decisions.md "Note
    Markdown rendering") are rendered to plain text before embedding, so raw syntax doesn't
    become part of the vector or leak into a cited RAG answer. Non-Markdown notes pass through
    unchanged, same as before this existed."""
    if not note.is_markdown:
        return note.content
    html = _markdown_to_html(note.content)
    # mistune's return type is a union because create_markdown() can be configured with an
    # AST renderer instead - _markdown_to_html above always uses the default "html" renderer,
    # so this is always a str in practice.
    assert isinstance(html, str)
    extractor = _HtmlTextExtractor()
    extractor.feed(html)
    return extractor.get_text()


def render_course(course: CourseDto) -> str:
    line2 = f"Semester {course.semester}, {course.ects} ECTS"
    if course.group:
        line2 += f", Elective group: {course.group}"
    lines = [f"Course: {course.name} ({course.code})", line2]
    if course.topics:
        lines.append(f"Topics: {', '.join(course.topics)}")
    return "\n".join(lines)


def render_session(session: StudySessionDto) -> str:
    lines = [
        f"Study session: {session.course_name}, "
        f"{session.start_time.strftime(DATETIME_FORMAT)} - "
        f"{session.end_time.strftime(DATETIME_FORMAT)}",
        f"Status: {'Completed' if session.is_completed else 'Planned'}",
    ]
    if session.topic:
        lines.append(f"Topic: {session.topic}")
    if session.notes:
        lines.append(f"Notes: {session.notes}")
    return "\n".join(lines)


def render_course_goal(goal: CourseGoalDto) -> str:
    lines = [f"Course goal: {goal.course_name}"]
    if goal.target_date:
        lines.append(f"Target date: {goal.target_date.strftime(_DATE_FORMAT)}")
    if goal.grade is not None:
        lines.append(f"Grade: {goal.grade}")
    if goal.completed_at:
        lines.append(f"Completed at: {goal.completed_at.strftime(DATETIME_FORMAT)}")
    if goal.completed_topics:
        lines.append(f"Completed topics: {goal.completed_topics}")
    if goal.completion_note:
        lines.append(f"Note: {goal.completion_note}")
    if goal.tag:
        lines.append(f"Tag: {goal.tag}")
    return "\n".join(lines)
