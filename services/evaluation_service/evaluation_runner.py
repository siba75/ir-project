import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dataset_manager import load_dataset_resources
from evaluation_checkpoints import checkpoint_path_for, load_checkpoint, save_checkpoint
from evaluation_config import DEFAULT_WORKERS, EVALUATION_RUNS, SEARCH_MODES, TOP_K, VECTOR_MODES
from evaluation_data import (
    build_pending_items,
    load_before_results_from_report,
    load_queries_and_qrels,
    select_query_ids,
)
from evaluation_report import build_comparison, summarize_run, write_report
from evaluation_search import run_evaluation_item


def run_evaluation(args, base_dir):
    queries, qrels = load_queries_and_qrels(base_dir, args.dataset)
    max_queries = None if args.all_queries or args.max_queries is None else args.max_queries
    query_ids = select_query_ids(qrels, max_queries)
    evaluation_scope = "all_qrels_queries" if max_queries is None else "sample"
    total_qrel_judgments = sum(len(doc_ids) for doc_ids in qrels.values())
    checkpoint_dir = Path(args.checkpoint_dir)

    print_evaluation_header(args, qrels, total_qrel_judgments, query_ids, evaluation_scope, checkpoint_dir)
    prepare_checkpoint_dir(checkpoint_dir, args.fresh)

    before_results = before_results_for(args, queries, qrels, query_ids, checkpoint_dir)
    after_results = {
        mode: evaluate_run(
            args.dataset,
            f"after_{mode}",
            queries,
            qrels,
            query_ids,
            checkpoint_dir,
            resume=not args.fresh,
            workers=args.workers,
            progress_every=args.progress_every,
            history_size=args.history_size,
        )
        for mode in SEARCH_MODES
    }

    output_path = Path(args.output) if args.output else base_dir / "reports" / f"{args.dataset}_evaluation_results.json"
    write_report(output_path, build_report(args, qrels, total_qrel_judgments, query_ids, evaluation_scope, before_results, after_results))
    print("Saved evaluation report to:", output_path)


def before_results_for(args, queries, qrels, query_ids, checkpoint_dir):
    if args.after_only:
        before_results = load_before_results_from_report(Path(args.reuse_before_report))
        print("Reused before-features results from:", args.reuse_before_report)
        return before_results

    return {
        mode: evaluate_run(
            args.dataset,
            f"before_{mode}",
            queries,
            qrels,
            query_ids,
            checkpoint_dir,
            resume=not args.fresh,
            workers=args.workers,
            progress_every=args.progress_every,
            history_size=args.history_size,
        )
        for mode in SEARCH_MODES
    }


def evaluate_run(
    dataset_name,
    run_name,
    queries,
    qrels,
    query_ids,
    checkpoint_dir,
    resume=True,
    workers=DEFAULT_WORKERS,
    progress_every=100,
    history_size=5,
):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_path_for(checkpoint_dir, dataset_name, run_name)
    completed_query_ids, sums = load_checkpoint(checkpoint_path, resume)
    completed_set = set(completed_query_ids)
    pending = build_pending_items(query_ids, queries, qrels, completed_set, history_size)

    print_run_header(dataset_name, run_name, completed_query_ids, pending, workers)
    warm_resources(dataset_name, run_name)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [
            executor.submit(
                run_evaluation_item,
                run_name,
                dataset_name,
                item["query_id"],
                item["query_text"],
                item["relevant_docs"],
                item["user_history"],
            )
            for item in pending
        ]
        collect_results(
            futures,
            checkpoint_path,
            dataset_name,
            run_name,
            len(query_ids),
            completed_query_ids,
            completed_set,
            sums,
            progress_every,
        )

    save_checkpoint(checkpoint_path, dataset_name, run_name, len(query_ids), completed_query_ids, sums)
    return summarize_run(dataset_name, run_name, completed_query_ids, sums)


def collect_results(
    futures,
    checkpoint_path,
    dataset_name,
    run_name,
    query_count,
    completed_query_ids,
    completed_set,
    sums,
    progress_every,
):
    for future in as_completed(futures):
        result = future.result()
        completed_query_ids.append(result["query_id"])
        completed_set.add(result["query_id"])

        for key in sums:
            sums[key] += result[key]

        completed_count = len(completed_query_ids)

        if completed_count == 1 or completed_count % progress_every == 0:
            print(f"[{completed_count}/{query_count}] {dataset_name}/{run_name}", flush=True)
            save_checkpoint(
                checkpoint_path,
                dataset_name,
                run_name,
                query_count,
                completed_query_ids,
                sums,
            )


def warm_resources(dataset_name, run_name):
    mode = EVALUATION_RUNS[run_name]["mode"]
    load_dataset_resources(dataset_name, include_vector=mode in VECTOR_MODES)


def build_report(args, qrels, total_qrel_judgments, query_ids, evaluation_scope, before_results, after_results):
    return {
        "dataset": args.dataset,
        "source_dataset": "beir/quora/test",
        "evaluation_scope": evaluation_scope,
        "total_qrels_queries": len(qrels),
        "total_qrel_judgments": total_qrel_judgments,
        "evaluated_queries": len(query_ids),
        "top_k": TOP_K,
        "workers": args.workers,
        "history_size": args.history_size,
        "before_features": before_results,
        "after_features": after_results,
        "comparison": build_comparison(before_results, after_results),
        "feature_notes": {
            "before_features": "Core retrieval modes without personalization.",
            "after_features": (
                "Personalization is applied to all modes using recent query history. "
                "Semantic and hybrid modes also demonstrate the FAISS vector store feature."
            ),
        },
    }


def prepare_checkpoint_dir(checkpoint_dir, fresh):
    if fresh and checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
        print("Deleted existing checkpoints because --fresh was used.")


def print_evaluation_header(args, qrels, total_qrel_judgments, query_ids, evaluation_scope, checkpoint_dir):
    print("Dataset:", args.dataset)
    print("Source dataset: beir/quora/test")
    print("Total qrels queries:", len(qrels))
    print("Total qrel judgments:", total_qrel_judgments)
    print("Queries selected for evaluation:", len(query_ids))
    print("Evaluation scope:", evaluation_scope)
    print("Workers:", args.workers)
    print("History size for personalization:", args.history_size)
    print("Checkpoint directory:", checkpoint_dir)


def print_run_header(dataset_name, run_name, completed_query_ids, pending, workers):
    print(
        f"Running {dataset_name}/{run_name}: "
        f"{len(completed_query_ids)} completed, {len(pending)} pending, workers={workers}",
        flush=True,
    )
