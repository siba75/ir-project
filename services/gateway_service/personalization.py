import logging
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


DEFAULT_PROFILE = {
    "enabled": False,
    "method": "TF-IDF history vector profile",
    "history_queries_used": 0,
    "interest_terms": [],
    "combined_terms": [],
    "similar_history_queries": [],
    "query_suggestions": [],
}


def clean_history_queries(user_history: list[str], current_query: str):
    cleaned_queries = []
    current_normalized = current_query.lower().strip()

    for query in user_history[-20:]:
        normalized_query = re.sub(r"\s+", " ", query.lower().strip())

        if not normalized_query or normalized_query == current_normalized:
            continue

        cleaned_queries.append(normalized_query)

    return cleaned_queries


def vector_top_terms(vector, feature_names, existing_terms, limit=6):
    dense_vector = np.asarray(vector).ravel()
    ranked_indices = dense_vector.argsort()[::-1]
    terms = []

    for index in ranked_indices:
        score = float(dense_vector[index])
        term = feature_names[index]

        if score <= 0:
            break

        term_words = set(term.split())

        if term in existing_terms or (term_words and term_words.issubset(existing_terms)):
            continue

        if len(term) < 3:
            continue

        terms.append({
            "term": term,
            "score": round(score, 6),
        })

        if len(terms) == limit:
            break

    return terms


def empty_profile():
    return dict(DEFAULT_PROFILE)


def build_personalized_query(refined_query: str, user_history: list[str]):
    history_queries = clean_history_queries(user_history, refined_query)

    if not history_queries:
        return refined_query, [], empty_profile()

    corpus = history_queries + [refined_query]

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=300,
            lowercase=True,
        )
        matrix = vectorizer.fit_transform(corpus)
    except ValueError as exc:
        logger.warning("Personalization skipped, vectorizer failed: %s", exc)
        return refined_query, [], empty_profile()

    history_matrix = matrix[:-1]
    query_vector = matrix[-1]
    feature_names = vectorizer.get_feature_names_out()
    existing_terms = set(re.findall(r"[a-zA-Z0-9]+", refined_query.lower()))

    similarities = (history_matrix @ query_vector.T).toarray().ravel()
    recency_weights = np.linspace(0.6, 1.0, num=len(history_queries))
    history_weights = (similarities + 0.08) * recency_weights
    weight_sum = float(history_weights.sum())

    if weight_sum == 0:
        history_weights = recency_weights
        weight_sum = float(history_weights.sum())

    interest_vector = (history_matrix.T @ history_weights) / weight_sum
    interest_vector = np.asarray(interest_vector).ravel()
    combined_vector = 0.70 * query_vector.toarray().ravel() + 0.30 * interest_vector

    interest_terms = vector_top_terms(
        interest_vector,
        feature_names,
        existing_terms,
        limit=8,
    )
    combined_terms = vector_top_terms(
        combined_vector,
        feature_names,
        existing_terms,
        limit=6,
    )
    selected_terms = select_personalization_terms(combined_terms, interest_terms)
    similar_history = find_similar_history(history_queries, similarities)
    query_suggestions = build_query_suggestions(refined_query, selected_terms, similar_history)
    personalized_query = refined_query

    if selected_terms:
        personalized_query = f"{refined_query} {' '.join(selected_terms)}"

    profile = {
        "enabled": True,
        "method": "TF-IDF history vector profile + query-interest vector fusion",
        "history_queries_used": len(history_queries),
        "query_vector_weight": 0.70,
        "interest_vector_weight": 0.30,
        "interest_terms": interest_terms,
        "combined_terms": combined_terms,
        "similar_history_queries": similar_history,
        "query_suggestions": query_suggestions,
    }

    return personalized_query, selected_terms, profile


def select_personalization_terms(combined_terms, interest_terms, limit=6):
    selected_terms = []

    for item in combined_terms + interest_terms:
        if item["term"] not in selected_terms:
            selected_terms.append(item["term"])

        if len(selected_terms) == limit:
            break

    return selected_terms


def find_similar_history(history_queries, similarities, limit=5):
    similar_indices = similarities.argsort()[::-1][:limit]
    similar_history = [
        {
            "query": history_queries[index],
            "similarity": round(float(similarities[index]), 6),
        }
        for index in similar_indices
        if similarities[index] > 0
    ]

    if not similar_history and history_queries:
        similar_history = [{
            "query": history_queries[-1],
            "similarity": 0.0,
        }]

    return similar_history


def build_query_suggestions(refined_query, selected_terms, similar_history):
    suggestions = [f"{refined_query} {term}" for term in selected_terms[:4]]

    for item in similar_history[:2]:
        if item["query"] not in suggestions:
            suggestions.append(item["query"])

    return suggestions[:6]
