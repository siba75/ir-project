from datasets import load_dataset
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

QUERIES_PATH = OUTPUT_DIR / "queries.json"
QRELS_PATH = OUTPUT_DIR / "qrels.json"


queries = {}
for item in queries_dataset["queries"]:
    queries[item["_id"]] = item["text"]



qrels = {}
for item in qrels_dataset["test"]:
    query_id = str(item["query-id"])
    doc_id = str(item["corpus-id"])
    score = int(item["score"])

    if score > 0:
        if query_id not in qrels:
            qrels[query_id] = []
        qrels[query_id].append(doc_id)

with open(QUERIES_PATH, "w", encoding="utf-8") as file:
    json.dump(queries, file, ensure_ascii=False, indent=2)

with open(QRELS_PATH, "w", encoding="utf-8") as file:
    json.dump(qrels, file, ensure_ascii=False, indent=2)

print("Queries:", len(queries))
print("Queries with qrels:", len(qrels))
print("Saved to:", OUTPUT_DIR)