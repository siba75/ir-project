import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download("stopwords", quiet=True)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    def fallback_lemma(token: str) -> str:
        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"
        suffixes = ("ches", "shes", "xes", "zes", "ses")
        if token.endswith(suffixes) and len(token) > 4:
            return token[:-2]
        if token.endswith("s") and len(token) > 3:
            return token[:-1]
        return token

    try:
        return [lemmatizer.lemmatize(token) for token in tokens]
    except Exception:
        return [fallback_lemma(token) for token in tokens]
