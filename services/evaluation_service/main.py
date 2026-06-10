from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import math

app = FastAPI(
    title="IR Evaluation Service",
    description="Service for evaluating retrieval models",
    version="1.0.0"
)


class EvaluationRequest(BaseModel):
    retrieved_docs: List[str]
    relevant_docs: List[str]
    k: int = 10


@app.get("/")
def home():
    return {
        "service": "Evaluation Service",
        "status": "running",
        "metrics": [
            "precision_at_k",
            "recall_at_k",
            "f1_score",
            "mrr",
            "average_precision"
        ]
    }


def precision_at_k(retrieved_docs, relevant_docs, k):
    retrieved_k = retrieved_docs[:k]

    if not retrieved_k:
        return 0.0

    relevant_retrieved = len(
        set(retrieved_k).intersection(set(relevant_docs))
    )

    return relevant_retrieved / len(retrieved_k)


def recall_at_k(retrieved_docs, relevant_docs, k):
    retrieved_k = retrieved_docs[:k]

    if not relevant_docs:
        return 0.0

    relevant_retrieved = len(
        set(retrieved_k).intersection(set(relevant_docs))
    )

    return relevant_retrieved / len(relevant_docs)


def f1_score(precision, recall):
    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def mean_reciprocal_rank(retrieved_docs, relevant_docs):
    for rank, doc_id in enumerate(retrieved_docs, start=1):
        if doc_id in relevant_docs:
            return 1 / rank

    return 0.0


def average_precision(retrieved_docs, relevant_docs):
    precisions = []
    relevant_found = 0

    for rank, doc_id in enumerate(retrieved_docs, start=1):
        if doc_id in relevant_docs:
            relevant_found += 1

            precision = relevant_found / rank
            precisions.append(precision)

    if not precisions:
        return 0.0

    return sum(precisions) / len(relevant_docs)


@app.post("/evaluate")
def evaluate(request: EvaluationRequest):

    if not request.retrieved_docs:
        raise HTTPException(
            status_code=400,
            detail="retrieved_docs cannot be empty"
        )

    if not request.relevant_docs:
        raise HTTPException(
            status_code=400,
            detail="relevant_docs cannot be empty"
        )

    precision = precision_at_k(
        request.retrieved_docs,
        request.relevant_docs,
        request.k
    )

    recall = recall_at_k(
        request.retrieved_docs,
        request.relevant_docs,
        request.k
    )

    f1 = f1_score(precision, recall)

    mrr = mean_reciprocal_rank(
        request.retrieved_docs,
        request.relevant_docs
    )

    ap = average_precision(
        request.retrieved_docs,
        request.relevant_docs
    )

    return {
        "k": request.k,
        "retrieved_documents": request.retrieved_docs,
        "relevant_documents": request.relevant_docs,
        "metrics": {
            "precision_at_k": round(precision, 4),
            "recall_at_k": round(recall, 4),
            "f1_score": round(f1, 4),
            "mrr": round(mrr, 4),
            "average_precision": round(ap, 4)
        }
    }