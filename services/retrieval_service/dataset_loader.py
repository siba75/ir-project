import ir_datasets

dataset = ir_datasets.load("cranfield")

print("Dataset Loaded Successfully")

print("\nFirst 3 Documents:\n")

for i, doc in enumerate(dataset.docs_iter()):
    if i >= 3:
        break

    print("DOC ID:", doc.doc_id)
    print("TEXT:", doc.text[:300])
    print("-" * 50)