import sys
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parents[1] / "services" / "preprocessing_service"
sys.path.insert(0, str(SERVICE_DIR))

from main import clean_text, lemmatize_tokens, preprocess_text, tokenize


def test_clean_text_lowercases():
    assert clean_text("Hello WORLD") == "hello world"


def test_clean_text_removes_urls():
    assert clean_text("visit https://example.com now") == "visit now"
    assert clean_text("go to www.site.org today") == "go to today"


def test_clean_text_removes_punctuation():
    assert clean_text("hello, world!") == "hello world"
    assert clean_text("a@b#c$d") == "a b c d"


def test_clean_text_collapses_whitespace():
    assert clean_text("  hello   world  ") == "hello world"


def test_clean_text_preserves_digits():
    assert clean_text("test 123 data") == "test 123 data"


def test_tokenize_splits_on_whitespace():
    assert tokenize("hello world") == ["hello", "world"]
    assert tokenize("one") == ["one"]
    assert tokenize("") == []


def test_lemmatize_tokens_basic():
    tokens = ["dogs", "cats", "running"]
    result = lemmatize_tokens(tokens)
    assert isinstance(result, list)
    assert len(result) == 3


def test_lemmatize_tokens_empty():
    assert lemmatize_tokens([]) == []


def test_preprocess_text_default_options():
    result = preprocess_text("Hello World! Visit https://example.com")
    assert result["original_text"] == "Hello World! Visit https://example.com"
    assert result["cleaned_text"] == "hello world visit"
    assert "processed_text" in result
    assert isinstance(result["tokens"], list)


def test_preprocess_text_removes_stopwords():
    result = preprocess_text("the quick brown fox is a dog", use_stemming=False)
    assert "the" not in result["tokens"]
    assert "is" not in result["tokens"]
    assert "a" not in result["tokens"]


def test_preprocess_text_no_stopword_removal():
    result = preprocess_text(
        "the dog is here",
        use_stemming=False,
        remove_stopwords=False,
    )
    assert "the" in result["tokens"]
    assert "is" in result["tokens"]


def test_preprocess_text_stemming():
    result = preprocess_text("running dogs jumping", use_stemming=True, remove_stopwords=False)
    assert all(isinstance(token, str) for token in result["tokens"])
    assert len(result["tokens"]) == 3


def test_preprocess_text_lemmatization():
    result = preprocess_text(
        "dogs cats",
        use_stemming=False,
        remove_stopwords=False,
        use_lemmatization=True,
    )
    assert isinstance(result["tokens"], list)
    assert len(result["tokens"]) == 2


def test_preprocess_text_processed_text_joins_tokens():
    result = preprocess_text("hello world", use_stemming=False, remove_stopwords=False)
    assert result["processed_text"] == " ".join(result["tokens"])
