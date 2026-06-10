import ir_datasets
import json
from pathlib import Path

DATASET_NAME = "msmarco-passage/train"
MAX_DOCS = 200000
MAX_QUERIES = 1000

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "datasets" / "msmarco"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DOCS_PATH = OUTPUT_DIR / "documents.json"
QUERIES_PATH = OUTPUT_DIR / "queries.json"
QRELS_PATH = OUTPUT_DIR / "qrels.json"


def main():
    print("Loading MS MARCO Passage dataset...")
    dataset = ir_datasets.load(DATASET_NAME)

    documents = {}
    queries = {}
    qrels = {}

    print(f"Loading first {MAX_DOCS} documents...")
    for index, doc in enumerate(dataset.docs_iter(), start=1):
        documents[doc.doc_id] = doc.text

        if index % 10000 == 0:
            print(f"Loaded documents: {index}")

        if index >= MAX_DOCS:
            break

    print(f"Loading first {MAX_QUERIES} queries...")
    for index, query in enumerate(dataset.queries_iter(), start=1):
        queries[query.query_id] = query.text

        if index >= MAX_QUERIES:
            break

    valid_doc_ids = set(documents.keys())
    valid_query_ids = set(queries.keys())

    print("Loading qrels...")
    for qrel in dataset.qrels_iter():
        query_id = qrel.query_id
        doc_id = qrel.doc_id

        if query_id in valid_query_ids and doc_id in valid_doc_ids:
            if query_id not in qrels:
                qrels[query_id] = []

            qrels[query_id].append(doc_id)

    with open(DOCS_PATH, "w", encoding="utf-8") as file:
        json.dump(documents, file, ensure_ascii=False)

    with open(QUERIES_PATH, "w", encoding="utf-8") as file:
        json.dump(queries, file, ensure_ascii=False)

    with open(QRELS_PATH, "w", encoding="utf-8") as file:
        json.dump(qrels, file, ensure_ascii=False)

    print("MS MARCO subset prepared successfully")
    print("Documents:", len(documents))
    print("Queries:", len(queries))
    print("Queries with qrels:", len(qrels))
    print("Saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()