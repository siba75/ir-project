import json


ZERO_SUMS = {
    "precision": 0.0,
    "recall": 0.0,
    "mrr": 0.0,
    "map": 0.0,
    "ndcg": 0.0,
}


def checkpoint_path_for(checkpoint_dir, dataset_name, run_name):
    return checkpoint_dir / f"{dataset_name}_{run_name}.json"


def load_checkpoint(path, resume):
    if not resume or not path.exists():
        return [], dict(ZERO_SUMS)

    with open(path, "r", encoding="utf-8") as file:
        state = json.load(file)

    sums = dict(ZERO_SUMS)
    sums.update(state.get("sums", {}))
    return state.get("completed_query_ids", []), sums


def save_checkpoint(path, dataset_name, run_name, query_count, completed_query_ids, sums):
    state = {
        "dataset": dataset_name,
        "run": run_name,
        "target_queries": query_count,
        "completed_queries": len(completed_query_ids),
        "completed_query_ids": completed_query_ids,
        "sums": sums,
    }

    with open(path, "w", encoding="utf-8") as file:
        json.dump(state, file, ensure_ascii=False, indent=2)
