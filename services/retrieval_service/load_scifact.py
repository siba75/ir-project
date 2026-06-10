from datasets import load_dataset
import pandas as pd
import os

print("Loading SciFact dataset...")

dataset = load_dataset("BeIR/scifact", "corpus")

docs = []

for item in dataset["corpus"]:
    docs.append({
        "doc_id": item["_id"],
        "text": item["title"] + " " + item["text"]
    })

df = pd.DataFrame(docs)

os.makedirs("../../datasets/scifact", exist_ok=True)

output_path = "../../datasets/scifact/scifact_docs.csv"

df.to_csv(output_path, index=False)

print(f"Saved {len(df)} documents")
print(f"Dataset saved to: {output_path}")