import argparse
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
RETRIEVAL_SERVICE_DIR = BASE_DIR / "services" / "retrieval_service"
GATEWAY_SERVICE_DIR = BASE_DIR / "services" / "gateway_service"
sys.path.insert(0, str(GATEWAY_SERVICE_DIR))
sys.path.insert(0, str(RETRIEVAL_SERVICE_DIR))

from evaluation_config import DEFAULT_WORKERS  # noqa: E402
from evaluation_runner import run_evaluation  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["quora"])
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Optional quick-test limit. Omit this to evaluate all qrels queries.",
    )
    parser.add_argument(
        "--all-queries",
        action="store_true",
        help="Evaluate all qrels queries. This is the required final mode.",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--history-size", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--after-only",
        action="store_true",
        help="Reuse existing all-query before-features results and evaluate only after-features.",
    )
    parser.add_argument(
        "--reuse-before-report",
        default=str(BASE_DIR / "reports" / "quora_evaluation_results.json"),
        help="Report JSON containing the all-query before-features results for all modes.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=str(BASE_DIR / "reports" / "evaluation_checkpoints"),
    )
    parser.add_argument("--fresh", action="store_true")
    return parser.parse_args()


def main():
    run_evaluation(parse_args(), BASE_DIR)


if __name__ == "__main__":
    main()
