from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re
import nltk
import importlib.resources as resources
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import PorterStemmer, WordNetLemmatizer
from symspellpy import SymSpell, Verbosity

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

app = FastAPI(
    title="IR Query Refinement Service",
    description="Service for query cleaning, spelling correction, WordNet query expansion, and refinement",
    version="2.0.0"
)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)


def load_symspell_dictionary() -> None:
    dictionary_file = resources.files("symspellpy").joinpath(
        "frequency_dictionary_en_82_765.txt"
    )

    with resources.as_file(dictionary_file) as dictionary_path:
        loaded = sym_spell.load_dictionary(
            str(dictionary_path),
            term_index=0,
            count_index=1
        )

    if not loaded:
        raise RuntimeError("Failed to load SymSpell frequency dictionary")


load_symspell_dictionary()


class QueryRefinementRequest(BaseModel):
    query: str
    remove_stopwords: bool = True
    use_stemming: bool = False
    use_lemmatization: bool = True
    use_expansion: bool = True
    use_spelling_correction: bool = True
    max_expansion_terms_per_token: int = 2


def clean_query(query: str) -> str:
    query = query.lower()
    query = re.sub(r"http\S+|www\S+", " ", query)
    query = re.sub(r"[^a-z0-9\s]", " ", query)
    query = re.sub(r"\s+", " ", query).strip()
    return query


def normalize_repeated_letters(token: str) -> str:
    """
    Reduce exaggerated repeated letters before spell correction.
    Example: howwww -> how, laaearn -> laearn
    """
    return re.sub(r"(.)\1{2,}", r"\1", token)


def should_skip_spelling_correction(token: str) -> bool:
    if not token:
        return True

    if token.isdigit():
        return True

    if len(token) <= 2:
        return True

    # Keep terms that contain numbers unchanged, مثل bm25 / ndcg10
    if any(char.isdigit() for char in token):
        return True

    return False


def correct_spelling_token(token: str) -> str:
    normalized_token = normalize_repeated_letters(token)

    if should_skip_spelling_correction(normalized_token):
        return normalized_token

    suggestions = sym_spell.lookup(
        normalized_token,
        Verbosity.CLOSEST,
        max_edit_distance=2,
        include_unknown=True
    )

    if not suggestions:
        return normalized_token

    return suggestions[0].term or normalized_token


def correct_spelling_tokens(tokens: list[str]) -> list[str]:
    return [correct_spelling_token(token) for token in tokens]


def get_wordnet_pos(token: str):
    """
    Simple POS heuristic for better lemmatization.
    """
    if token.endswith("ing") or token.endswith("ed"):
        return wn.VERB
    if token.endswith("ly"):
        return wn.ADV
    if token.endswith("ous") or token.endswith("ful") or token.endswith("able") or token.endswith("ive"):
        return wn.ADJ
    return wn.NOUN


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    lemmatized_tokens = []

    for token in tokens:
        pos = get_wordnet_pos(token)
        lemma = lemmatizer.lemmatize(token, pos=pos)
        lemmatized_tokens.append(lemma)

    return lemmatized_tokens


def get_wordnet_expansion_terms(token: str, max_terms: int = 2) -> list[str]:
    """
    Get professional query expansion terms using WordNet.

    Filtering rules:
    - no stopwords
    - no multi-word phrases
    - no original token repeated
    - alphabetic terms only
    - limited number of terms to avoid noisy queries
    """
    if not token:
        return []

    if token in stop_words:
        return []

    if not token.isalpha():
        return []

    if len(token) <= 2:
        return []

    expansion_terms = []

    for synset in wn.synsets(token):
        for lemma in synset.lemmas():
            candidate = lemma.name().lower().replace("_", " ")

            if " " in candidate:
                continue

            if not candidate.isalpha():
                continue

            if candidate == token:
                continue

            if candidate in stop_words:
                continue

            if candidate not in expansion_terms:
                expansion_terms.append(candidate)

            if len(expansion_terms) >= max_terms:
                return expansion_terms

    return expansion_terms


def expand_query_tokens(tokens: list[str], max_terms_per_token: int = 2) -> list[str]:
    expanded_terms = []

    for token in tokens:
        expanded_terms.extend(
            get_wordnet_expansion_terms(
                token=token,
                max_terms=max_terms_per_token
            )
        )

    return expanded_terms


def remove_duplicate_terms(tokens: list[str]) -> list[str]:
    seen = set()
    unique_tokens = []

    for token in tokens:
        if token not in seen:
            unique_tokens.append(token)
            seen.add(token)

    return unique_tokens


def refine_query(
    query: str,
    remove_stopwords: bool = True,
    use_stemming: bool = False,
    use_lemmatization: bool = True,
    use_expansion: bool = True,
    use_spelling_correction: bool = True,
    max_expansion_terms_per_token: int = 2
):
    cleaned_query = clean_query(query)

    if not cleaned_query:
        raise HTTPException(status_code=400, detail="Query is empty after cleaning")

    tokens = cleaned_query.split()

    if use_spelling_correction:
        tokens = correct_spelling_tokens(tokens)

    corrected_query = " ".join(tokens)

    if remove_stopwords:
        tokens = [token for token in tokens if token not in stop_words]

    if use_lemmatization:
        tokens = lemmatize_tokens(tokens)

    if use_stemming:
        tokens = [stemmer.stem(token) for token in tokens]

    expanded_terms = []

    if use_expansion:
        safe_max_terms = max(0, min(max_expansion_terms_per_token, 3))
        expanded_terms = expand_query_tokens(
            tokens=tokens,
            max_terms_per_token=safe_max_terms
        )

    final_tokens = tokens + expanded_terms
    final_tokens = remove_duplicate_terms(final_tokens)

    return {
        "original_query": query,
        "cleaned_query": cleaned_query,
        "corrected_query": corrected_query,
        "tokens": tokens,
        "expanded_terms": expanded_terms,
        "use_spelling_correction": use_spelling_correction,
        "use_lemmatization": use_lemmatization,
        "use_expansion": use_expansion,
        "max_expansion_terms_per_token": max_expansion_terms_per_token,
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
        use_expansion=request.use_expansion,
        use_spelling_correction=request.use_spelling_correction,
        max_expansion_terms_per_token=request.max_expansion_terms_per_token
    )