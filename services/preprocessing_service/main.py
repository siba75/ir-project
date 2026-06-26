import logging
import re

import nltk
from fastapi import FastAPI
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from pydantic import BaseModel

logger = logging.getLogger(__name__)

nltk.download("stopwords", quiet=True)

app = FastAPI(
    title="IR Preprocessing Service",
    description="Service for cleaning and preprocessing documents and queries",
    version="1.0.0"
)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


class TextRequest(BaseModel):
    text: str
    use_stemming: bool = True
    use_lemmatization: bool = False
    remove_stopwords: bool = True


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    return text.split()


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    def fallback_lemma(token: str) -> str:
        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"
        if token.endswith(("ches", "shes", "xes", "zes", "ses")) and len(token) > 4:
            return token[:-2]
        if token.endswith("s") and len(token) > 3:
            return token[:-1]
        return token

    try:
        return [lemmatizer.lemmatize(token) for token in tokens]
    except LookupError as exc:
        logger.warning("WordNet resource missing, using fallback lemmatizer: %s", exc)
        return [fallback_lemma(token) for token in tokens]


def preprocess_text(
    text: str,
    use_stemming: bool = True,
    remove_stopwords: bool = True,
    use_lemmatization: bool = False
):
    cleaned_text = clean_text(text)
    tokens = tokenize(cleaned_text)

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
