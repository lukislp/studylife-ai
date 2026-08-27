"""Shared prompt-injection defense: escaping untrusted text before it's embedded in a prompt
sent to the LLM.

Used by both `rag/prompt.py` (the `<notes>` DATA block `/chat` builds from retrieved chunks)
and `agent/tools.py` (the `search_notes` tool's output, fed straight into `/agent`'s
LangChain tool loop). Both wrap the same underlying untrusted source - a note, which can
originate from studylife-capture's arbitrary-webpage ingestion (see docs/decisions.md
"Capture enrichment") - so this is one implementation instead of two near-identical copies.
"""


def escape_untrusted_text(text: str) -> str:
    """Neutralize literal `<`/`>` so untrusted content can't fake a closing tag or boundary
    marker and break out of whatever DATA framing wraps it (prompt-injection defense) - the
    content is untrusted as far as prompt structure is concerned, even when it's the user's
    own data, since it could contain text pasted or captured from anywhere."""
    return text.replace("<", "&lt;").replace(">", "&gt;")
