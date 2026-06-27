import json

from evaluation_config import EVALUATION_RUNS, TOP_K


def summarize_run(dataset_name, run_name, completed_query_ids, sums):
    total = len(completed_query_ids)
    run_config = EVALUATION_RUNS[run_name]

    return {
        "dataset": dataset_name,
        "retrieval_mode": run_config["mode"],
        "phase": run_config["phase"],
        "evaluated_queries": total,
        f"mean_precision@{TOP_K}": round(sums["precision"] / total, 4) if total else 0,
        f"mean_recall@{TOP_K}": round(sums["recall"] / total, 4) if total else 0,
        "mrr": round(sums["mrr"] / total, 4) if total else 0,
        "map": round(sums["map"] / total, 4) if total else 0,
        f"ndcg@{TOP_K}": round(sums["ndcg"] / total, 4) if total else 0,
        "description": run_config["description"],
    }


def build_comparison(before_results, after_results):
    comparison = {}

    for mode, before_result in before_results.items():
        after_result = after_results.get(mode, {})
        comparison[mode] = metric_delta(before_result, after_result)

    return comparison


def metric_delta(before_result, after_result):
    return {
        "precision@10_delta": round(
            after_result.get(f"mean_precision@{TOP_K}", 0)
            - before_result.get(f"mean_precision@{TOP_K}", 0),
            4,
        ),
        "recall@10_delta": round(
            after_result.get(f"mean_recall@{TOP_K}", 0)
            - before_result.get(f"mean_recall@{TOP_K}", 0),
            4,
        ),
        "mrr_delta": round(after_result.get("mrr", 0) - before_result.get("mrr", 0), 4),
        "map_delta": round(after_result.get("map", 0) - before_result.get("map", 0), 4),
        f"ndcg@{TOP_K}_delta": round(
            after_result.get(f"ndcg@{TOP_K}", 0)
            - before_result.get(f"ndcg@{TOP_K}", 0),
            4,
        ),
    }


def write_report(output_path, report):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
