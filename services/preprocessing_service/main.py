from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download("stopwords", quiet=True)

app = FastAPI(
    title="IR Preprocessing Service",
    description="Service for cleaning and preprocessing documents and queries",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8501", "http://localhost:8501"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


MAX_TEXT_LENGTH = 50000


class TextRequest(BaseModel):
    text: str
    use_stemming: bool = True
    use_lemmatization: bool = False
    remove_stopwords: bool = True

    @field_validator("text")
    @classmethod
    def text_not_too_long(cls, v: str) -> str:
        if len(v) > MAX_TEXT_LENGTH:
            raise ValueError(f"text exceeds maximum length of {MAX_TEXT_LENGTH} characters")
        return v


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
    except Exception:
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
