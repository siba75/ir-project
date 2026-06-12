import sys
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parents[1] / "services" / "evaluation_service"
sys.path.insert(0, str(SERVICE_DIR))

from metrics import (
    average_precision,
    f1_score,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)


def test_precision_recall_f1():
    retrieved = ["d1", "d2", "d3"]
    relevant = ["d2", "d4"]

    precision = precision_at_k(retrieved, relevant, 3)
    recall = recall_at_k(retrieved, relevant, 3)

    assert precision == 1 / 3
    assert recall == 1 / 2
    assert round(f1_score(precision, recall), 4) == 0.4


def test_mrr_and_average_precision():
    retrieved = ["d3", "d2", "d1", "d4"]
    relevant = ["d1", "d4"]

    assert mean_reciprocal_rank(retrieved, relevant) == 1 / 3
    assert round(average_precision(retrieved, relevant), 4) == 0.4167
