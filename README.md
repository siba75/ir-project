# Information Retrieval System 2026

Service-oriented Information Retrieval system built on `beir/quora/test` from `ir-datasets`.

## Dataset

- Dataset: `beir/quora/test`
- Documents: 522,931
- Queries: 10,000
- Qrels: available for evaluation

This project is configured for one approved dataset, based on the instructor's approval that one dataset is acceptable if it satisfies the requirements.

## Services

- `preprocessing_service`: text normalization, stopword removal, stemming.
- `indexing_service`: inverted-index API.
- `retrieval_service`: TF-IDF, BM25, FAISS semantic search, hybrid parallel, hybrid serial.
- `refinement_service`: query cleaning and synonym expansion.
- `evaluation_service`: IR metrics and Quora benchmark evaluation.
- `gateway_service`: orchestrates query refinement and retrieval.
- `ui/streamlit_app`: Streamlit interface.

Architecture and design-pattern documentation:

```text
docs/SOA_DESIGN_PATTERNS_AR.md
```

Evaluation-requirements checklist:

```text
docs/EVALUATION_REQUIREMENTS_AR.md
```

Indexing, query processing, refinement, and ranking checklist:

```text
docs/IR_CORE_REQUIREMENTS_AR.md
```

Preprocessing and representation checklist:

```text
docs/PREPROCESSING_REPRESENTATION_REQUIREMENTS_AR.md
```

## Run

```bat
run_services.bat
```

Open:

```text
http://localhost:8501
```

## Evaluation

```bash
python services/evaluation_service/evaluate_dataset_direct.py quora --all-queries
```

The generated report includes before/after feature tables and is saved to:

```text
reports/quora_evaluation_results.json
```

## Retrieval Models

- TF-IDF
- BM25 with prebuilt `k1=1.5` and `b=0.75`
- Semantic FAISS vector search. The current Quora index uses `sentence_transformer`; the builder also supports `lsa_tfidf_svd`.
- Hybrid Parallel score fusion
- Hybrid Serial BM25 candidate generation plus semantic reranking

## Additional Features

- FAISS vector store
- Query refinement / expansion
- Query personalization from search history
- Result clustering in the UI
