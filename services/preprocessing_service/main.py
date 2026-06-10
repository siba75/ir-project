from fastapi import FastAPI
from pydantic import BaseModel
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords", quiet=True)

app = FastAPI(
    title="IR Preprocessing Service",
    description="Service for cleaning and preprocessing documents and queries",
    version="1.0.0"
)

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


class TextRequest(BaseModel):
    text: str
    use_stemming: bool = True
    remove_stopwords: bool = True


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    return text.split()


def preprocess_text(text: str, use_stemming: bool = True, remove_stopwords: bool = True):
    cleaned_text = clean_text(text)
    tokens = tokenize(cleaned_text)

    if remove_stopwords:
        tokens = [token for token in tokens if token not in stop_words]

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
        request.remove_stopwords
    )
    result["type"] = "document"
    return result


@app.post("/preprocess/query")
def preprocess_query(request: TextRequest):
    result = preprocess_text(
        request.text,
        request.use_stemming,
        request.remove_stopwords
    )
    result["type"] = "query"
    return result