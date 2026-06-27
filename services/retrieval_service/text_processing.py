import sys
from pathlib import Path

_SHARED_DIR = str(Path(__file__).resolve().parent.parent / "shared")
if _SHARED_DIR not in sys.path:
    sys.path.append(_SHARED_DIR)

from text_cleaning import clean_text  # noqa: E402


DOCUMENT_TEXT_FIELDS = [
    "title",
    "text",
    "body",
    "abstract",
    "summary",
    "detailed_description",
    "brief_title",
    "brief_summary",
    "condition",
    "intervention",
    "author",
]


def preprocess_text(text: str) -> list[str]:
    return clean_text(text).split()


def field_to_text(value):
    if value is None:
        return ""

    if isinstance(value, (list, tuple)):
        return " ".join(field_to_text(item) for item in value)

    return str(value)


def doc_to_text(doc):
    parts = []

    for field_name in DOCUMENT_TEXT_FIELDS:
        if hasattr(doc, field_name):
            value = field_to_text(getattr(doc, field_name))

            if value:
                parts.append(value)

    if not parts:
        parts.append(str(doc))

    return " ".join(parts)


def query_to_text(query):
    for field_name in ["text", "title", "description", "query"]:
        if hasattr(query, field_name):
            value = field_to_text(getattr(query, field_name))

            if value:
                return value

    return str(query)


def iter_batches(items, batch_size=10000):
    batch = []

    for item in items:
        batch.append(item)

        if len(batch) == batch_size:
            yield batch
            batch = []

    if batch:
        yield batch
