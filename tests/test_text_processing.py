import sys
from pathlib import Path
from types import SimpleNamespace

SERVICE_DIR = Path(__file__).resolve().parents[1] / "services" / "retrieval_service"
sys.path.insert(0, str(SERVICE_DIR))

from text_processing import (
    DOCUMENT_TEXT_FIELDS,
    doc_to_text,
    field_to_text,
    iter_batches,
    preprocess_text,
    query_to_text,
)


def test_preprocess_text_lowercases_and_tokenizes():
    assert preprocess_text("Hello World") == ["hello", "world"]


def test_preprocess_text_removes_urls():
    tokens = preprocess_text("visit https://example.com now")
    assert "https" not in " ".join(tokens)
    assert "visit" in tokens
    assert "now" in tokens


def test_preprocess_text_removes_punctuation():
    assert preprocess_text("hello, world!") == ["hello", "world"]


def test_preprocess_text_collapses_whitespace():
    assert preprocess_text("  hello   world  ") == ["hello", "world"]


def test_preprocess_text_preserves_digits():
    assert preprocess_text("test 42") == ["test", "42"]


def test_preprocess_text_empty_string():
    assert preprocess_text("") == []


def test_field_to_text_none():
    assert field_to_text(None) == ""


def test_field_to_text_string():
    assert field_to_text("hello") == "hello"


def test_field_to_text_number():
    assert field_to_text(42) == "42"


def test_field_to_text_list():
    assert field_to_text(["hello", "world"]) == "hello world"


def test_field_to_text_nested_list():
    assert field_to_text(["a", ["b", "c"]]) == "a b c"


def test_field_to_text_tuple():
    assert field_to_text(("x", "y")) == "x y"


def test_field_to_text_list_with_none():
    assert field_to_text(["hello", None, "world"]) == "hello  world"


def test_doc_to_text_uses_known_fields():
    doc = SimpleNamespace(text="doc text", title="doc title")
    result = doc_to_text(doc)
    assert "doc title" in result
    assert "doc text" in result


def test_doc_to_text_falls_back_to_str():
    doc = SimpleNamespace(unknown_field="irrelevant")
    result = doc_to_text(doc)
    assert "irrelevant" in result


def test_doc_to_text_skips_missing_fields():
    doc = SimpleNamespace(title="only title")
    result = doc_to_text(doc)
    assert result == "only title"


def test_query_to_text_uses_text_field():
    query = SimpleNamespace(text="search query")
    assert query_to_text(query) == "search query"


def test_query_to_text_uses_title_field():
    query = SimpleNamespace(title="query title")
    assert query_to_text(query) == "query title"


def test_query_to_text_prefers_text_over_title():
    query = SimpleNamespace(text="primary", title="secondary")
    assert query_to_text(query) == "primary"


def test_query_to_text_falls_back_to_str():
    query = SimpleNamespace(unknown="value")
    result = query_to_text(query)
    assert "unknown" in result or "value" in result


def test_query_to_text_uses_query_field():
    query = SimpleNamespace(query="my query")
    assert query_to_text(query) == "my query"


def test_iter_batches_single_batch():
    items = [1, 2, 3]
    batches = list(iter_batches(items, batch_size=10))
    assert batches == [[1, 2, 3]]


def test_iter_batches_exact_split():
    items = [1, 2, 3, 4]
    batches = list(iter_batches(items, batch_size=2))
    assert batches == [[1, 2], [3, 4]]


def test_iter_batches_remainder():
    items = [1, 2, 3, 4, 5]
    batches = list(iter_batches(items, batch_size=2))
    assert batches == [[1, 2], [3, 4], [5]]


def test_iter_batches_empty():
    batches = list(iter_batches([], batch_size=5))
    assert batches == []


def test_iter_batches_default_size():
    items = list(range(25000))
    batches = list(iter_batches(items))
    assert len(batches) == 3
    assert len(batches[0]) == 10000
    assert len(batches[1]) == 10000
    assert len(batches[2]) == 5000


def test_document_text_fields_is_list():
    assert isinstance(DOCUMENT_TEXT_FIELDS, list)
    assert "text" in DOCUMENT_TEXT_FIELDS
    assert "title" in DOCUMENT_TEXT_FIELDS
