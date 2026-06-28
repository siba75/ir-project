import argparse
from pathlib import Path

from ir_resource_builder import build_resources, build_vectors_only


DATASET_CONFIGS = {
    "quora": "beir/quora/test",
}

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BASE_DIR = Path(__file__).resolve().parents[2]
DATASETS_DIR = BASE_DIR / "datasets"
INDEXES_DIR = BASE_DIR / "indexes"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        choices=sorted(DATASET_CONFIGS.keys()) + ["all"],
    )
    parser.add_argument(
        "--skip-vectors",
        action="store_true",
        help="Build only lexical resources. Use this for quick download/index checks.",
    )
    parser.add_argument(
        "--vectors-only",
        action="store_true",
        help="Build only FAISS resources from existing BM25 documents.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--vector-method",
        choices=["lsa", "transformer"],
        default="transformer",
        help="Use lsa for fast dense TF-IDF+SVD embeddings or transformer for SentenceTransformer embeddings.",
    )
    return parser.parse_args()


def selected_aliases(dataset_argument):
    return DATASET_CONFIGS.keys() if dataset_argument == "all" else [dataset_argument]


def main():
    args = parse_args()

    for alias in selected_aliases(args.dataset):
        dataset_id = DATASET_CONFIGS[alias]
        output_dataset_dir = DATASETS_DIR / alias
        output_index_dir = INDEXES_DIR / alias

        if args.vectors_only:
            build_vectors_only(
                alias=alias,
                dataset_id=dataset_id,
                model_name=MODEL_NAME,
                output_index_dir=output_index_dir,
                batch_size=args.batch_size,
                vector_method=args.vector_method,
            )
            continue

        build_resources(
            alias=alias,
            dataset_id=dataset_id,
            model_name=MODEL_NAME,
            output_dataset_dir=output_dataset_dir,
            output_index_dir=output_index_dir,
            build_vectors=not args.skip_vectors,
            batch_size=args.batch_size,
            vector_method=args.vector_method,
        )


if __name__ == "__main__":
    main()
