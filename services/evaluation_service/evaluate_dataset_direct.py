import argparse
import json
import shutil
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
RETRIEVAL_SERVICE_DIR = BASE_DIR / "services" / "retrieval_service"
_SHARED_DIR = str(Path(__file__).resolve().parent.parent / "shared")
sys.path.insert(0, str(RETRIEVAL_SERVICE_DIR))
if _SHARED_DIR not in sys.path:
    sys.path.append(_SHARED_DIR)

from main import (  # noqa: E402
    dataset_bm25_search,
    dataset_tfidf_search,
    hybrid_search,
    hybrid_serial_search,
    semantic_search,
)
from metrics import (  # noqa: E402
    average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from synonyms import SYNONYMS  # noqa: E402


TOP_K = 10
EVALUATION_MODES = [
    "tfidf",
    "bm25",
    "semantic",
    "hybrid_parallel",
    "hybrid_serial",
]


def refine_query_for_after_features(query):
    terms = query.lower().split()
    expanded_terms = []

    for term in terms:
        clean_term = "".join(ch for ch in term if ch.isalnum())

        if clean_term in SYNONYMS:
            expanded_terms.extend(SYNONYMS[clean_term])

    combined_terms = terms + expanded_terms[:8]
    return " ".join(combined_terms)


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
    use_after_features=False,
    progress_every=100,
    checkpoint_dir=None,
    resume=True
):
    phase = "after_features" if use_after_features else "before_features"
    checkpoint_path = None
    completed_query_ids = []
    completed_set = set()
    sums = {
        "precision": 0.0,
        "recall": 0.0,
        "mrr": 0.0,
        "map": 0.0,
        "ndcg": 0.0,
    }

    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{dataset_name}_{phase}_{mode}.json"

        if resume and checkpoint_path.exists():
            with open(checkpoint_path, "r", encoding="utf-8") as file:
                state = json.load(file)

            completed_query_ids = state.get("completed_query_ids", [])
            completed_set = set(completed_query_ids)
            sums.update(state.get("sums", {}))

            print(
                f"Resuming {dataset_name}/{phase}/{mode}: "
                f"{len(completed_query_ids)}/{len(query_ids)} queries already done",
                flush=True
            )

    def save_checkpoint():
        if not checkpoint_path:
            return

        state = {
            "dataset": dataset_name,
            "retrieval_mode": mode,
            "phase": phase,
            "target_queries": len(query_ids),
            "completed_queries": len(completed_query_ids),
            "completed_query_ids": completed_query_ids,
            "sums": sums,
        }

        with open(checkpoint_path, "w", encoding="utf-8") as file:
            json.dump(state, file, ensure_ascii=False, indent=2)

    for counter, query_id in enumerate(query_ids, start=1):
        if query_id in completed_set:
            continue

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

        sums["precision"] += precision_at_k(retrieved_docs, relevant_docs, TOP_K)
        sums["recall"] += recall_at_k(retrieved_docs, relevant_docs, TOP_K)
        sums["mrr"] += mean_reciprocal_rank(retrieved_docs, relevant_docs)
        sums["map"] += average_precision(retrieved_docs, relevant_docs)
        sums["ndcg"] += ndcg_at_k(retrieved_docs, relevant_docs, TOP_K)

        completed_query_ids.append(query_id)
        completed_set.add(query_id)
        completed_count = len(completed_query_ids)
        phase_label = "after" if use_after_features else "before"

        if (
            completed_count == 1
            or completed_count % progress_every == 0
            or completed_count == len(query_ids)
        ):
            print(
                f"[{completed_count}/{len(query_ids)}] "
                f"{dataset_name}/{phase_label}/{mode}/{query_id}",
                flush=True
            )
            save_checkpoint()

    save_checkpoint()
    total = len(completed_query_ids)

    return {
        "dataset": dataset_name,
        "retrieval_mode": mode,
        "phase": phase,
        "evaluated_queries": total,
        f"mean_precision@{TOP_K}": round(sums["precision"] / total, 4) if total else 0,
        f"mean_recall@{TOP_K}": round(sums["recall"] / total, 4) if total else 0,
        "mrr": round(sums["mrr"] / total, 4) if total else 0,
        "map": round(sums["map"] / total, 4) if total else 0,
        f"ndcg@{TOP_K}": round(sums["ndcg"] / total, 4) if total else 0,
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
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print one progress line every N evaluated queries."
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=str(BASE_DIR / "reports" / "evaluation_checkpoints"),
        help="Directory used to save/resume evaluation checkpoints."
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete existing checkpoints before starting."
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
    checkpoint_dir = Path(args.checkpoint_dir)
    print("Checkpoint directory:", checkpoint_dir)

    if args.fresh and checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
        print("Deleted existing checkpoints because --fresh was used.")

    before_results = {
        mode: evaluate_mode(
            args.dataset,
            mode,
            queries,
            qrels,
            query_ids,
            use_after_features=False,
            progress_every=args.progress_every,
            checkpoint_dir=checkpoint_dir,
            resume=not args.fresh
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
            use_after_features=True,
            progress_every=args.progress_every,
            checkpoint_dir=checkpoint_dir,
            resume=not args.fresh
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
