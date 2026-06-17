import argparse
import json
import math
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
RETRIEVAL_SERVICE_DIR = BASE_DIR / "services" / "retrieval_service"
sys.path.insert(0, str(RETRIEVAL_SERVICE_DIR))

from main import (  # noqa: E402
    dataset_bm25_search,
    dataset_tfidf_search,
    hybrid_search,
    hybrid_serial_search,
    semantic_search,
)


TOP_K = 10
EVALUATION_MODES = [
    "tfidf",
    "bm25",
    "semantic",
    "hybrid_parallel",
    "hybrid_serial",
]

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
}


def refine_query_for_after_features(query):
    terms = query.lower().split()
    expanded_terms = []

    for term in terms:
        clean_term = "".join(ch for ch in term if ch.isalnum())

        if clean_term in SYNONYMS:
            expanded_terms.extend(SYNONYMS[clean_term])

    combined_terms = terms + expanded_terms[:8]
    return " ".join(combined_terms)


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


def run_search(mode, query, dataset_name):
    if mode == "tfidf":
        return dataset_tfidf_search(query, TOP_K, dataset_name)

    if mode == "bm25":
        return dataset_bm25_search(query, TOP_K, dataset_name, 1.5, 0.75)

    if mode == "semantic":
        return semantic_search(query, TOP_K, dataset_name)

    if mode == "hybrid_parallel":
        return hybrid_search(query, TOP_K, 0.4, 0.6, dataset_name)

    if mode == "hybrid_serial":
        return hybrid_serial_search(query, TOP_K, 50, dataset_name)

    raise ValueError(f"Unsupported mode: {mode}")


def load_queries_and_qrels(dataset_name):
    dataset_dir = BASE_DIR / "datasets" / dataset_name

    with open(dataset_dir / "queries.json", "r", encoding="utf-8") as file:
        queries = json.load(file)

    with open(dataset_dir / "qrels.json", "r", encoding="utf-8") as file:
        qrels = json.load(file)

    return queries, qrels


def select_query_ids(qrels, max_queries=None):
    query_ids = list(qrels.keys())

    if max_queries is None:
        return query_ids

    return query_ids[:max_queries]


def evaluate_mode(
    dataset_name,
    mode,
    queries,
    qrels,
    query_ids,
    use_after_features=False
):
    precision_scores = []
    recall_scores = []
    mrr_scores = []
    map_scores = []
    ndcg_scores = []

    for counter, query_id in enumerate(query_ids, start=1):
        query_text = queries.get(query_id)
        relevant_docs = qrels.get(query_id, [])

        if not query_text or not relevant_docs:
            continue

        effective_query = (
            refine_query_for_after_features(query_text)
            if use_after_features
            else query_text
        )

        data = run_search(mode, effective_query, dataset_name)
        retrieved_docs = [
            str(item["doc_id"])
            for item in data.get("results", [])
        ]

        precision_scores.append(precision_at_k(retrieved_docs, relevant_docs, TOP_K))
        recall_scores.append(recall_at_k(retrieved_docs, relevant_docs, TOP_K))
        mrr_scores.append(reciprocal_rank(retrieved_docs, relevant_docs))
        map_scores.append(average_precision(retrieved_docs, relevant_docs))
        ndcg_scores.append(ndcg_at_k(retrieved_docs, relevant_docs, TOP_K))

        phase = "after" if use_after_features else "before"
        print(f"[{counter}] {dataset_name}/{phase}/{mode}/{query_id}")

    total = len(precision_scores)

    return {
        "dataset": dataset_name,
        "retrieval_mode": mode,
        "phase": "after_features" if use_after_features else "before_features",
        "evaluated_queries": total,
        f"mean_precision@{TOP_K}": round(sum(precision_scores) / total, 4) if total else 0,
        f"mean_recall@{TOP_K}": round(sum(recall_scores) / total, 4) if total else 0,
        "mrr": round(sum(mrr_scores) / total, 4) if total else 0,
        "map": round(sum(map_scores) / total, 4) if total else 0,
        f"ndcg@{TOP_K}": round(sum(ndcg_scores) / total, 4) if total else 0,
    }


def build_comparison(before_results, after_results):
    comparison = {}

    for mode, before_values in before_results.items():
        after_values = after_results.get(mode, {})

        comparison[mode] = {
            "precision@10_delta": round(
                after_values.get(f"mean_precision@{TOP_K}", 0)
                - before_values.get(f"mean_precision@{TOP_K}", 0),
                4
            ),
            "recall@10_delta": round(
                after_values.get(f"mean_recall@{TOP_K}", 0)
                - before_values.get(f"mean_recall@{TOP_K}", 0),
                4
            ),
            "mrr_delta": round(
                after_values.get("mrr", 0) - before_values.get("mrr", 0),
                4
            ),
            "map_delta": round(
                after_values.get("map", 0) - before_values.get("map", 0),
                4
            ),
            f"ndcg@{TOP_K}_delta": round(
                after_values.get(f"ndcg@{TOP_K}", 0)
                - before_values.get(f"ndcg@{TOP_K}", 0),
                4
            ),
        }

    return comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["quora"])
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Optional quick-test limit. Omit this to evaluate all qrels queries."
    )
    parser.add_argument(
        "--all-queries",
        action="store_true",
        help="Evaluate all qrels queries. This can take many hours on Quora."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSON path. Defaults to reports/<dataset>_evaluation_results.json."
    )
    args = parser.parse_args()

    queries, qrels = load_queries_and_qrels(args.dataset)
    max_queries = None if args.all_queries or args.max_queries is None else args.max_queries
    query_ids = select_query_ids(qrels, max_queries)
    evaluation_scope = "all_qrels_queries" if max_queries is None else "sample"
    total_qrel_judgments = sum(len(doc_ids) for doc_ids in qrels.values())

    print("Dataset:", args.dataset)
    print("Source dataset: beir/quora/test")
    print("Total qrels queries:", len(qrels))
    print("Total qrel judgments:", total_qrel_judgments)
    print("Queries selected for evaluation:", len(query_ids))
    print("Evaluation scope:", evaluation_scope)

    before_results = {
        mode: evaluate_mode(
            args.dataset,
            mode,
            queries,
            qrels,
            query_ids,
            use_after_features=False
        )
        for mode in EVALUATION_MODES
    }

    after_results = {
        mode: evaluate_mode(
            args.dataset,
            mode,
            queries,
            qrels,
            query_ids,
            use_after_features=True
        )
        for mode in EVALUATION_MODES
    }

    all_results = {
        "dataset": args.dataset,
        "source_dataset": "beir/quora/test",
        "evaluation_scope": evaluation_scope,
        "total_qrels_queries": len(qrels),
        "total_qrel_judgments": total_qrel_judgments,
        "evaluated_queries": len(query_ids),
        "top_k": TOP_K,
        "before_features": before_results,
        "after_features": after_results,
        "comparison": build_comparison(before_results, after_results),
        "feature_notes": {
            "before_features": "Raw benchmark queries with core retrieval models.",
            "after_features": "Queries expanded with the query refinement feature. FAISS vector store is used by semantic and hybrid models. UI personalization and result clustering are interactive features and are documented separately."
        }
    }

    output_path = (
        Path(args.output)
        if args.output
        else BASE_DIR / "reports" / f"{args.dataset}_evaluation_results.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(all_results, file, ensure_ascii=False, indent=2)

    print("Saved evaluation report to:", output_path)


if __name__ == "__main__":
    main()
