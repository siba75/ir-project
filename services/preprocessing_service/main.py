import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

_SHARED_DIR = str(Path(__file__).resolve().parent.parent / "shared")
if _SHARED_DIR not in sys.path:
    sys.path.append(_SHARED_DIR)

from nlp_utils import lemmatize_tokens, stemmer, stop_words  # noqa: E402
from text_cleaning import clean_text  # noqa: E402

app = FastAPI(
    title="IR Preprocessing Service",
    description="Service for cleaning and preprocessing documents and queries",
    version="1.0.0"
)


class TextRequest(BaseModel):
    text: str
    use_stemming: bool = True
    use_lemmatization: bool = False
    remove_stopwords: bool = True


def preprocess_text(
    text: str,
    use_stemming: bool = True,
    remove_stopwords: bool = True,
    use_lemmatization: bool = False
):
    cleaned_text = clean_text(text)
    tokens = cleaned_text.split()

    if remove_stopwords:
        tokens = [token for token in tokens if token not in stop_words]

    if use_lemmatization:
        tokens = lemmatize_tokens(tokens)

    if use_stemming:
        tokens = [stemmer.stem(token) for token in tokens]

    return {
        "original_text": text,
        "cleaned_text": cleaned_text,
        "tokens": tokens,
        "processed_text": " ".join(tokens)
    }


@app.get("/")
def home():
    return {
        "service": "Preprocessing Service",
        "status": "running"
    }


@app.post("/preprocess/document")
def preprocess_document(request: TextRequest):
    result = preprocess_text(
        request.text,
        request.use_stemming,
        request.remove_stopwords,
        request.use_lemmatization
    )
    result["type"] = "document"
    return result


@app.post("/preprocess/query")
def preprocess_query(request: TextRequest):
    result = preprocess_text(
        request.text,
        request.use_stemming,
        request.remove_stopwords,
        request.use_lemmatization
    )
    result["type"] = "query"
    return result
