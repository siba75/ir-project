import pickle
from collections import Counter

import ir_datasets
from rank_bm25 import BM25Okapi

from ir_dataset_utils import get_doc_id, get_query_id, save_json
from text_processing import doc_to_text, iter_batches, preprocess_text, query_to_text
from vector_builders import build_lsa_vectors, build_transformer_vectors


def build_resources(
    alias,
    dataset_id,
    model_name,
    output_dataset_dir,
    output_index_dir,
    build_vectors=True,
    batch_size=64,
    vector_method="lsa",
):
    dataset = ir_datasets.load(dataset_id)
    output_dataset_dir.mkdir(parents=True, exist_ok=True)
    output_index_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading documents for {alias}: {dataset_id}")
    doc_ids, documents, tokenized_documents, inverted_index, document_lengths = load_documents(dataset)
    queries = load_queries(dataset)
    qrels = load_qrels(dataset)

    save_json(output_dataset_dir / "queries.json", queries)
    save_json(output_dataset_dir / "qrels.json", qrels)
    save_index_json(output_index_dir, alias, dataset_id, doc_ids, documents, inverted_index, document_lengths)
    save_bm25(output_index_dir, documents, doc_ids, tokenized_documents)

    if build_vectors:
        build_vectors_for_method(
            alias,
            dataset_id,
            model_name,
            documents,
            doc_ids,
            output_index_dir,
            vector_method,
            batch_size,
        )

    print_summary(alias, dataset_id, documents, queries, qrels)


def build_vectors_only(alias, dataset_id, model_name, output_index_dir, batch_size=64, vector_method="lsa"):
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
    build_vectors_for_method(
        alias,
        dataset_id,
        model_name,
        documents,
        doc_ids,
        output_index_dir,
        vector_method,
        batch_size,
    )
    print("Vector resources saved")


def load_documents(dataset):
    doc_ids = []
    documents = []
    inverted_index = {}
    document_lengths = {}
    tokenized_documents = []
    loaded_documents = 0

    for batch in iter_batches(dataset.docs_iter(), batch_size=10000):
        for doc in batch:
            add_document(doc, doc_ids, documents, tokenized_documents, inverted_index, document_lengths)

        loaded_documents += len(batch)
        print(f"Loaded documents: {loaded_documents}")

    return doc_ids, documents, tokenized_documents, inverted_index, document_lengths


def add_document(doc, doc_ids, documents, tokenized_documents, inverted_index, document_lengths):
    doc_id = get_doc_id(doc)
    text = doc_to_text(doc)
    tokens = preprocess_text(text)

    doc_ids.append(doc_id)
    documents.append(text)
    document_lengths[doc_id] = len(tokens)
    tokenized_documents.append(tokens)

    for term, frequency in Counter(tokens).items():
        inverted_index.setdefault(term, {})[doc_id] = frequency


def load_queries(dataset):
    print("Loading queries")
    return {
        get_query_id(query): query_to_text(query)
        for query in dataset.queries_iter()
    }


def load_qrels(dataset):
    print("Loading qrels")
    qrels = {}

    for qrel in dataset.qrels_iter():
        relevance = getattr(qrel, "relevance", 1)

        if relevance <= 0:
            continue

        query_id = str(qrel.query_id)
        doc_id = str(qrel.doc_id)
        qrels.setdefault(query_id, []).append(doc_id)

    return qrels


def save_index_json(output_index_dir, alias, dataset_id, doc_ids, documents, inverted_index, document_lengths):
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


def save_bm25(output_index_dir, documents, doc_ids, tokenized_documents):
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


def build_vectors_for_method(alias, dataset_id, model_name, documents, doc_ids, output_index_dir, vector_method, batch_size):
    if vector_method == "transformer":
        build_transformer_vectors(
            alias,
            dataset_id,
            model_name,
            documents,
            doc_ids,
            output_index_dir,
            batch_size,
        )
        return

    build_lsa_vectors(
        alias,
        dataset_id,
        documents,
        doc_ids,
        output_index_dir,
    )


def print_summary(alias, dataset_id, documents, queries, qrels):
    print("Done")
    print("Dataset:", alias)
    print("Source:", dataset_id)
    print("Documents:", len(documents))
    print("Queries:", len(queries))
    print("Queries with qrels:", len(qrels))
