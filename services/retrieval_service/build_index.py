import ir_datasets
import json
import re
from collections import defaultdict, Counter
from pathlib import Path


DATASET_NAME = "cranfield"
MAX_DOCUMENTS = None

BASE_DIR = Path(__file__).resolve().parents[2]
INDEX_DIR = BASE_DIR / "indexes"
INDEX_DIR.mkdir(exist_ok=True)

INDEX_PATH = INDEX_DIR / "cranfield_inverted_index.json"


def preprocess(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


def build_index():
    dataset = ir_datasets.load(DATASET_NAME)

    inverted_index = defaultdict(dict)
    documents_store = {}
    document_lengths = {}

    for i, doc in enumerate(dataset.docs_iter()):
        if MAX_DOCUMENTS and i >= MAX_DOCUMENTS:
            break

        tokens = preprocess(doc.text)
        term_frequencies = Counter(tokens)

        documents_store[doc.doc_id] = doc.text
        document_lengths[doc.doc_id] = len(tokens)

        for term, frequency in term_frequencies.items():
            inverted_index[term][doc.doc_id] = frequency

    index_data = {
        "dataset": DATASET_NAME,
        "total_documents": len(documents_store),
        "unique_terms": len(inverted_index),
        "documents_store": documents_store,
        "document_lengths": document_lengths,
        "inverted_index": dict(inverted_index)
    }

    with open(INDEX_PATH, "w", encoding="utf-8") as file:
        json.dump(index_data, file, ensure_ascii=False)

    print("Index built successfully")
    print("Dataset:", DATASET_NAME)
    print("Total documents:", len(documents_store))
    print("Unique terms:", len(inverted_index))
    print("Saved to:", INDEX_PATH)


if __name__ == "__main__":
    build_index()