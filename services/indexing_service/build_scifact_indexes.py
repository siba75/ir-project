import os
import pickle
import pandas as pd
import faiss
import numpy as np

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer



documents = df["text"].fillna("").tolist()
doc_ids = df["doc_id"].astype(str).tolist()

print(f"Loaded {len(documents)} documents")

# =====================================================
# BM25 INDEX
# =====================================================

print("Building BM25 index...")

tokenized_docs = [doc.lower().split() for doc in documents]

bm25 = BM25Okapi(tokenized_docs)



print("BM25 index saved")

# =====================================================
# FAISS VECTOR INDEX
# =====================================================

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings...")

embeddings = model.encode(
    documents,
    show_progress_bar=True,
    convert_to_numpy=True
)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)





print("FAISS index saved")

