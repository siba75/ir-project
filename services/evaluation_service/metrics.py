import math


def precision_at_k(retrieved_docs, relevant_docs, k):
    retrieved_k = retrieved_docs[:k]

    if not retrieved_k:
        return 0.0

    relevant_retrieved = len(
        set(retrieved_k).intersection(set(relevant_docs))
    )

    return relevant_retrieved / len(retrieved_k)


def recall_at_k(retrieved_docs, relevant_docs, k):
    retrieved_k = retrieved_docs[:k]

    if not relevant_docs:
        return 0.0

    relevant_retrieved = len(
        set(retrieved_k).intersection(set(relevant_docs))
    )

    return relevant_retrieved / len(relevant_docs)


def f1_score(precision, recall):
    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def mean_reciprocal_rank(retrieved_docs, relevant_docs):
    for rank, doc_id in enumerate(retrieved_docs, start=1):
        if doc_id in relevant_docs:
            return 1 / rank

    return 0.0


def average_precision(retrieved_docs, relevant_docs):
    precisions = []
    relevant_found = 0

    for rank, doc_id in enumerate(retrieved_docs, start=1):
        if doc_id in relevant_docs:
            relevant_found += 1
            precision = relevant_found / rank
            precisions.append(precision)

    if not precisions:
        return 0.0

    return sum(precisions) / len(relevant_docs)


def dcg_at_k(retrieved_docs, relevant_docs, k):
    relevant_set = set(relevant_docs)
    score = 0.0

    for rank, doc_id in enumerate(retrieved_docs[:k], start=1):
        if doc_id in relevant_set:
            score += 1 / math.log2(rank + 1)

    return score


def ndcg_at_k(retrieved_docs, relevant_docs, k):
    if not relevant_docs:
        return 0.0

    ideal_hits = min(len(relevant_docs), k)

    if ideal_hits == 0:
        return 0.0

    ideal_dcg = sum(
        1 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    if ideal_dcg == 0:
        return 0.0

    return dcg_at_k(retrieved_docs, relevant_docs, k) / ideal_dcg
