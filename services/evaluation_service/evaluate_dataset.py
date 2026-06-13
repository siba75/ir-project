import argparse
import json
import math
import requests
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
GATEWAY_URL = "http://127.0.0.1:8006/search/full"

TOP_K = 10
EVALUATION_MODES = [
    "tfidf",
    "bm25",
    "semantic",
    "hybrid_parallel",
    "hybrid_serial",
]


def precision_at_k(retrieved, relevant, k):
    return len(set(retrieved[:k]) & set(relevant)) / k


def recall_at_k(retrieved, relevant, k):
    if not relevant:
        return 0
    return len(set(retrieved[:k]) & set(relevant)) / len(relevant)


def reciprocal_rank(retrieved, relevant):
    relevant = set(relevant)

    for index, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1 / index

    return 0


def average_precision(retrieved, relevant):
    relevant = set(relevant)

    if not relevant:
        return 0

    hits = 0
    score = 0

    for index, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            hits += 1
            score += hits / index

    return score / len(relevant)


def dcg_at_k(retrieved, relevant, k):
    relevant = set(relevant)
    score = 0

    for index, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            score += 1 / math.log2(index + 1)

    return score


def ndcg_at_k(retrieved, relevant, k):
    ideal_hits = min(len(relevant), k)

    if ideal_hits == 0:
        return 0

    ideal_dcg = sum(
        1 / math.log2(index + 1)
        for index in range(1, ideal_hits + 1)
    )

    return dcg_at_k(retrieved, relevant, k) / ideal_dcg


def evaluate_mode(dataset_name, mode, max_queries):
    dataset_dir = BASE_DIR / "datasets" / dataset_name

    with open(dataset_dir / "queries.json", "r", encoding="utf-8") as file:
        queries = json.load(file)

    with open(dataset_dir / "qrels.json", "r", encoding="utf-8") as file:
        qrels = json.load(file)

    precision_scores = []
    recall_scores = []
    mrr_scores = []
    map_scores = []
    ndcg_scores = []

    query_ids = list(qrels.keys())[:max_queries]

    for counter, query_id in enumerate(query_ids, start=1):
        query_text = queries.get(query_id)
        relevant_docs = qrels.get(query_id, [])

        if not query_text or not relevant_docs:
            continue

        payload = {
            "query": query_text,
            "top_k": TOP_K,
            "dataset": dataset_name,
            "retrieval_mode": mode,
            "bm25_weight": 0.4,
            "semantic_weight": 0.6,
            "bm25_k1": 1.5,
            "bm25_b": 0.75,
            "initial_k": 50,
            "remove_stopwords": True,
            "use_stemming": False,
            "use_expansion": False,
            "use_personalization": False,
        }

        response = requests.post(GATEWAY_URL, json=payload, timeout=180)

        if response.status_code != 200:
            print(f"[{counter}] {dataset_name}/{mode}/{query_id} failed: {response.text}")
            continue

        data = response.json()
        retrieved_docs = [
            str(item["doc_id"])
            for item in data.get("results", [])
        ]

        precision_scores.append(precision_at_k(retrieved_docs, relevant_docs, TOP_K))
        recall_scores.append(recall_at_k(retrieved_docs, relevant_docs, TOP_K))
        mrr_scores.append(reciprocal_rank(retrieved_docs, relevant_docs))
        map_scores.append(average_precision(retrieved_docs, relevant_docs))
        ndcg_scores.append(ndcg_at_k(retrieved_docs, relevant_docs, TOP_K))

        print(f"[{counter}] {dataset_name}/{mode}/{query_id}")

    total = len(precision_scores)

    return {
        "dataset": dataset_name,
        "retrieval_mode": mode,
        "evaluated_queries": total,
        f"mean_precision@{TOP_K}": round(sum(precision_scores) / total, 4) if total else 0,
        f"mean_recall@{TOP_K}": round(sum(recall_scores) / total, 4) if total else 0,
        "mrr": round(sum(mrr_scores) / total, 4) if total else 0,
        "map": round(sum(map_scores) / total, 4) if total else 0,
        f"ndcg@{TOP_K}": round(sum(ndcg_scores) / total, 4) if total else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["quora"])
    parser.add_argument("--max-queries", type=int, default=200)
    args = parser.parse_args()

    all_results = {
        mode: evaluate_mode(args.dataset, mode, args.max_queries)
        for mode in EVALUATION_MODES
    }

    output_path = BASE_DIR / "reports" / f"{args.dataset}_gateway_evaluation_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(all_results, file, ensure_ascii=False, indent=2)

    print("Saved evaluation report to:", output_path)


if __name__ == "__main__":
    main()
