import argparse
import json
import pickle
import re
from collections import Counter
from pathlib import Path

import faiss
import ir_datasets
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


DATASET_CONFIGS = {
    "quora": "beir/quora/test",
    
}

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BASE_DIR = Path(__file__).resolve().parents[2]
DATASETS_DIR = BASE_DIR / "datasets"
INDEXES_DIR = BASE_DIR / "indexes"


def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


def field_to_text(value):
    if value is None:
        return ""

    if isinstance(value, (list, tuple)):
        return " ".join(field_to_text(item) for item in value)

    return str(value)


def doc_to_text(doc):
    parts = []

    for field_name in [
        "title",
        "text",
        "body",
        "abstract",
        "summary",
        "detailed_description",
        "brief_title",
        "brief_summary",
        "condition",
        "intervention",
    ]:
        if hasattr(doc, field_name):
            value = field_to_text(getattr(doc, field_name))
            if value:
                parts.append(value)

    if not parts:
        parts.append(str(doc))

    return " ".join(parts)


def query_to_text(query):
    for field_name in ["text", "title", "description", "query"]:
        if hasattr(query, field_name):
            value = field_to_text(getattr(query, field_name))
            if value:
                return value

    return str(query)


def get_query_id(query):
    for field_name in ["query_id", "qid"]:
        if hasattr(query, field_name):
            return str(getattr(query, field_name))

    raise ValueError(f"Cannot identify query id for {query}")


def get_doc_id(doc):
    for field_name in ["doc_id", "docno"]:
        if hasattr(doc, field_name):
            return str(getattr(doc, field_name))

    raise ValueError(f"Cannot identify document id for {doc}")


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False)


def build_lsa_vectors(alias, documents, doc_ids, output_index_dir, n_components=256):
    print("Building LSA dense vector index")
    print("TF-IDF max_features: 50000")
    print("SVD components:", n_components)

    vectorizer = TfidfVectorizer(
        max_features=50000,
        stop_words="english",
        lowercase=True,
    )
    tfidf_matrix = vectorizer.fit_transform(documents)

    svd = TruncatedSVD(
        n_components=n_components,
        random_state=42,
    )
    dense_vectors = svd.fit_transform(tfidf_matrix)
    dense_vectors = normalize(dense_vectors).astype("float32")

    dimension = dense_vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(dense_vectors)
    faiss.write_index(index, str(output_index_dir / "faiss.index"))

    metadata = {
        "dataset": alias,
        "source_dataset": DATASET_CONFIGS[alias],
        "vector_method": "lsa_tfidf_svd",
        "doc_ids": doc_ids,
        "documents": documents,
        "total_documents": len(documents),
        "embedding_dimension": int(dimension),
        "vectorizer": vectorizer,
        "svd": svd,
    }

    with open(output_index_dir / "metadata.pkl", "wb") as file:
        pickle.dump(metadata, file)


def build_transformer_vectors(alias, documents, doc_ids, output_index_dir, batch_size=64):
    print("Building SentenceTransformer FAISS vector index")
    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        documents,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    faiss.write_index(index, str(output_index_dir / "faiss.index"))

    metadata = {
        "dataset": alias,
        "source_dataset": DATASET_CONFIGS[alias],
        "vector_method": "sentence_transformer",
        "model_name": MODEL_NAME,
        "doc_ids": doc_ids,
        "documents": documents,
        "total_documents": len(documents),
        "embedding_dimension": int(dimension),
    }

    with open(output_index_dir / "metadata.pkl", "wb") as file:
        pickle.dump(metadata, file)


def build_resources(alias, build_vectors=True, batch_size=64, vector_method="lsa"):
    dataset_id = DATASET_CONFIGS[alias]
    dataset = ir_datasets.load(dataset_id)

    output_dataset_dir = DATASETS_DIR / alias
    output_index_dir = INDEXES_DIR / alias
    output_dataset_dir.mkdir(parents=True, exist_ok=True)
    output_index_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading documents for {alias}: {dataset_id}")

    doc_ids = []
    documents = []
    inverted_index = {}
    document_lengths = {}
    tokenized_documents = []

    for index, doc in enumerate(dataset.docs_iter(), start=1):
        doc_id = get_doc_id(doc)
        text = doc_to_text(doc)
        tokens = preprocess(text)

        doc_ids.append(doc_id)
        documents.append(text)
        document_lengths[doc_id] = len(tokens)
        tokenized_documents.append(tokens)

        for term, frequency in Counter(tokens).items():
            if term not in inverted_index:
                inverted_index[term] = {}
            inverted_index[term][doc_id] = frequency

        if index % 10000 == 0:
            print(f"Loaded documents: {index}")

    print("Loading queries")
    queries = {
        get_query_id(query): query_to_text(query)
        for query in dataset.queries_iter()
    }

    print("Loading qrels")
    qrels = {}

    for qrel in dataset.qrels_iter():
        relevance = getattr(qrel, "relevance", 1)

        if relevance <= 0:
            continue

        query_id = str(qrel.query_id)
        doc_id = str(qrel.doc_id)
        qrels.setdefault(query_id, []).append(doc_id)

    save_json(output_dataset_dir / "queries.json", queries)
    save_json(output_dataset_dir / "qrels.json", qrels)

    index_data = {
        "dataset": alias,
        "source_dataset": dataset_id,
        "total_documents": len(documents),
        "unique_terms": len(inverted_index),
        "documents_store": dict(zip(doc_ids, documents)),
        "document_lengths": document_lengths,
        "inverted_index": inverted_index,
    }

    save_json(output_index_dir / "inverted_index.json", index_data)

    print("Building BM25")
    bm25 = BM25Okapi(tokenized_documents)

    with open(output_index_dir / "bm25.pkl", "wb") as file:
        pickle.dump(
            {
                "bm25": bm25,
                "documents": documents,
                "doc_ids": doc_ids,
            },
            file,
        )

    if build_vectors:
        if vector_method == "transformer":
            build_transformer_vectors(alias, documents, doc_ids, output_index_dir, batch_size)
        else:
            build_lsa_vectors(alias, documents, doc_ids, output_index_dir)

    print("Done")
    print("Dataset:", alias)
    print("Source:", dataset_id)
    print("Documents:", len(documents))
    print("Queries:", len(queries))
    print("Queries with qrels:", len(qrels))


def build_vectors_only(alias, batch_size=64, vector_method="lsa"):
    dataset_id = DATASET_CONFIGS[alias]
    output_index_dir = INDEXES_DIR / alias
    bm25_path = output_index_dir / "bm25.pkl"

    if not bm25_path.exists():
        raise FileNotFoundError(
            f"BM25 resources not found at {bm25_path}. Run without --vectors-only first."
        )

    with open(bm25_path, "rb") as file:
        bm25_data = pickle.load(file)

    documents = bm25_data["documents"]
    doc_ids = bm25_data["doc_ids"]

    print("Dataset:", alias)
    print("Documents:", len(documents))

    if vector_method == "transformer":
        build_transformer_vectors(alias, documents, doc_ids, output_index_dir, batch_size)
    else:
        build_lsa_vectors(alias, documents, doc_ids, output_index_dir)

    print("Vector resources saved")


def main():
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
        default="lsa",
        help="Use lsa for fast dense TF-IDF+SVD embeddings or transformer for SentenceTransformer embeddings.",
    )
    args = parser.parse_args()

    aliases = DATASET_CONFIGS.keys() if args.dataset == "all" else [args.dataset]

    for alias in aliases:
        if args.vectors_only:
            build_vectors_only(
                alias=alias,
                batch_size=args.batch_size,
                vector_method=args.vector_method,
            )
        else:
            build_resources(
                alias=alias,
                build_vectors=not args.skip_vectors,
                batch_size=args.batch_size,
                vector_method=args.vector_method,
            )


if __name__ == "__main__":
    main()
