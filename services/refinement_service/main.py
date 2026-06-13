from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download("stopwords", quiet=True)

app = FastAPI(
    title="IR Query Refinement Service",
    description="Service for query cleaning, normalization, expansion, and refinement",
    version="1.0.0"
)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

SYNONYMS = {
    "learn": ["study", "practice"],
    "learning": ["studying", "training"],
    "programming": ["coding", "development"],
    "code": ["programming", "software"],
    "job": ["career", "work"],
    "best": ["top", "recommended"],
    "good": ["best", "useful"],
    "difference": ["comparison", "compare"],
    "start": ["begin", "learn"],
    "language": ["programming"],
    "computer": ["technology", "software"],
    "business": ["company", "startup"],
    "money": ["finance", "income"],
    "health": ["medical", "wellness"],
    "phone": ["mobile", "smartphone"],
    "india": ["indian"],
    "usa": ["america", "american"],
    "retrieval": ["search", "ranking"],
    "information": ["data", "knowledge"],
    "engine": ["system", "machine"],
    "model": ["method", "approach", "technique"]
}


class QueryRefinementRequest(BaseModel):
    query: str
    remove_stopwords: bool = True
    use_stemming: bool = False
    use_lemmatization: bool = False
    use_expansion: bool = True


def clean_query(query: str) -> str:
    query = query.lower()
    query = re.sub(r"http\S+|www\S+", " ", query)
    query = re.sub(r"[^a-z0-9\s]", " ", query)
    query = re.sub(r"\s+", " ", query).strip()
    return query


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


def refine_query(
    query: str,
    remove_stopwords: bool = True,
    use_stemming: bool = False,
    use_lemmatization: bool = False,
    use_expansion: bool = True
):
    cleaned_query = clean_query(query)

    if not cleaned_query:
        raise HTTPException(status_code=400, detail="Query is empty after cleaning")

    tokens = cleaned_query.split()

    if remove_stopwords:
        tokens = [token for token in tokens if token not in stop_words]

    expanded_terms = []

    if use_expansion:
        for token in tokens:
            if token in SYNONYMS:
                expanded_terms.extend(SYNONYMS[token])

    final_tokens = tokens + expanded_terms

    if use_lemmatization:
        final_tokens = lemmatize_tokens(final_tokens)

    if use_stemming:
        final_tokens = [stemmer.stem(token) for token in final_tokens]

    return {
        "original_query": query,
        "cleaned_query": cleaned_query,
        "tokens": tokens,
        "expanded_terms": expanded_terms,
        "use_lemmatization": use_lemmatization,
        "refined_query": " ".join(final_tokens)
    }


@app.get("/")
def home():
    return {
        "service": "Query Refinement Service",
        "status": "running"
    }


@app.post("/refine")
def refine(request: QueryRefinementRequest):
    return refine_query(
        query=request.query,
        remove_stopwords=request.remove_stopwords,
        use_stemming=request.use_stemming,
        use_lemmatization=request.use_lemmatization,
        use_expansion=request.use_expansion
    )
