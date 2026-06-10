import ir_datasets
import requests
from collections import defaultdict
from statistics import mean

GATEWAY_URL = "http://127.0.0.1:8006/search/full"

TOP_K = 10
MAX_QUERIES = 50


def precision_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]

    if not retrieved_k:
        return 0

    relevant_count = sum(1 for doc_id in retrieved_k if doc_id in relevant)

    return relevant_count / k


def recall_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]

    if not relevant:
        return 0

    relevant_count = sum(1 for doc_id in retrieved_k if doc_id in relevant)

    return relevant_count / len(relevant)


def reciprocal_rank(retrieved, relevant):
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1 / rank

    return 0


def average_precision(retrieved, relevant):
    precisions = []
    relevant_found = 0

    for rank, doc_id in enumerate(retrieved, start=1):

        if doc_id in relevant:
            relevant_found += 1

            precision = relevant_found / rank

            precisions.append(precision)

    if not precisions:
        return 0

    return sum(precisions) / len(relevant)


def load_qrels(dataset):
    qrels = defaultdict(set)

    for qrel in dataset.qrels_iter():

        if qrel.relevance > 0:
            qrels[qrel.query_id].add(qrel.doc_id)

    return qrels


def evaluate():

    print("Loading Cranfield dataset...")

    dataset = ir_datasets.load("cranfield")

    queries = list(dataset.queries_iter())

    qrels = load_qrels(dataset)

    precision_scores = []
    recall_scores = []
    mrr_scores = []
    map_scores = []

    evaluated_queries = 0

    for query in queries[:MAX_QUERIES]:

        query_id = query.query_id
        query_text = query.text

        relevant_docs = qrels.get(query_id, set())

        if not relevant_docs:
            continue

        payload = {
            "query": query_text,
            "top_k": TOP_K,
            "bm25_weight": 0.4,
            "semantic_weight": 0.6,
            "remove_stopwords": True,
            "use_stemming": False,
            "use_expansion": True
        }

        try:

            response = requests.post(
                GATEWAY_URL,
                json=payload,
                timeout=120
            )

            if response.status_code != 200:
                print(f"Query {query_id} failed")
                continue

            data = response.json()

            retrieved_docs = [
                result["doc_id"]
                for result in data["results"]
            ]

            precision = precision_at_k(
                retrieved_docs,
                relevant_docs,
                TOP_K
            )

            recall = recall_at_k(
                retrieved_docs,
                relevant_docs,
                TOP_K
            )

            rr = reciprocal_rank(
                retrieved_docs,
                relevant_docs
            )

            ap = average_precision(
                retrieved_docs,
                relevant_docs
            )

            precision_scores.append(precision)
            recall_scores.append(recall)
            mrr_scores.append(rr)
            map_scores.append(ap)

            evaluated_queries += 1

            print(f"[{evaluated_queries}] Query ID: {query_id}")
            print(f"Precision@{TOP_K}: {precision:.4f}")
            print(f"Recall@{TOP_K}: {recall:.4f}")
            print(f"MRR: {rr:.4f}")
            print(f"AP: {ap:.4f}")
            print("-" * 50)

        except Exception as error:
            print(f"Error for query {query_id}: {error}")

    print("\n========== FINAL RESULTS ==========")

    print(f"Evaluated Queries: {evaluated_queries}")

    print(f"Mean Precision@{TOP_K}: {mean(precision_scores):.4f}")

    print(f"Mean Recall@{TOP_K}: {mean(recall_scores):.4f}")

    print(f"MRR: {mean(mrr_scores):.4f}")

    print(f"MAP: {mean(map_scores):.4f}")


if __name__ == "__main__":
    evaluate()