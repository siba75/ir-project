import pickle

import faiss
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from text_processing import preprocess_text


def build_lsa_vectors(
    alias,
    source_dataset,
    documents,
    doc_ids,
    output_index_dir,
    n_components=256,
):
    print("Building LSA dense vector index")
    print("TF-IDF max_features: 50000")
    print("SVD components:", n_components)

    vectorizer = TfidfVectorizer(
        max_features=50000,
        analyzer=preprocess_text,
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
        "source_dataset": source_dataset,
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


def build_transformer_vectors(
    alias,
    source_dataset,
    model_name,
    documents,
    doc_ids,
    output_index_dir,
    batch_size=64,
):
    print("Building SentenceTransformer FAISS vector index")
    model = SentenceTransformer(model_name)

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
        "source_dataset": source_dataset,
        "vector_method": "sentence_transformer",
        "vector_method_alias": "transformer",
        "model_name": model_name,
        "doc_ids": doc_ids,
        "documents": documents,
        "total_documents": len(documents),
        "embedding_dimension": int(dimension),
    }

    with open(output_index_dir / "metadata.pkl", "wb") as file:
        pickle.dump(metadata, file)
