import ir_datasets
import faiss
import numpy as np
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer


DATASET_NAME = "cranfield"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BASE_DIR = Path(__file__).resolve().parents[2]
INDEX_DIR = BASE_DIR / "indexes"
INDEX_DIR.mkdir(exist_ok=True)

FAISS_INDEX_PATH = INDEX_DIR / "cranfield_faiss.index"
METADATA_PATH = INDEX_DIR / "cranfield_vector_metadata.pkl"


def build_vector_index():
    print("Loading dataset...")
    dataset = ir_datasets.load(DATASET_NAME)

    documents = []
    doc_ids = []

    for doc in dataset.docs_iter():
        doc_ids.append(doc.doc_id)
        documents.append(doc.text)

    print(f"Loaded {len(documents)} documents")

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Encoding documents...")
    embeddings = model.encode(
        documents,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]

    print("Building FAISS index...")
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    print("Saving FAISS index...")
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    metadata = {
        "dataset": DATASET_NAME,
        "model_name": MODEL_NAME,
        "doc_ids": doc_ids,
        "documents": documents,
        "total_documents": len(documents),
        "embedding_dimension": dimension
    }

    with open(METADATA_PATH, "wb") as file:
        pickle.dump(metadata, file)

    print("Vector index built successfully")
    print("Dataset:", DATASET_NAME)
    print("Total documents:", len(documents))
    print("Embedding dimension:", dimension)
    print("FAISS index saved to:", FAISS_INDEX_PATH)
    print("Metadata saved to:", METADATA_PATH)


if __name__ == "__main__":
    build_vector_index()