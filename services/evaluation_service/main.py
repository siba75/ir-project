from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from metrics import (
    average_precision,
    f1_score,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

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
            "average_precision",
            "ndcg_at_k"
        ]
    }


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

    ndcg = ndcg_at_k(
        request.retrieved_docs,
        request.relevant_docs,
        request.k
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
            "average_precision": round(ap, 4),
            "ndcg_at_k": round(ndcg, 4)
        }
    }
