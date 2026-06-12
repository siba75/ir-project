# Information Retrieval System 2026

Python implementation of a service-oriented Information Retrieval system.

## Architecture

The project is split into independent services:

- `services/preprocessing_service`: text cleaning, normalization, stopword removal, stemming.
- `services/indexing_service`: in-memory inverted index API.
- `services/retrieval_service`: TF-IDF, BM25, semantic FAISS search, hybrid parallel search, hybrid serial search.
- `services/refinement_service`: query cleaning and synonym-based query expansion.
- `services/evaluation_service`: IR metrics and benchmark evaluation scripts.
- `services/gateway_service`: API gateway that coordinates refinement and retrieval.
- `ui/streamlit_app`: Streamlit user interface.

Communication between services is done through REST APIs.

## Current Datasets

The current local build contains:

- Cranfield
- SciFact

Important: the assignment requires two datasets with more than 200K documents and qrels. The current datasets are useful for development and demo, but they must be replaced or extended with approved large datasets before final submission.

Recommended candidates from `ir-datasets`:

- `msmarco-passage/train`
- another approved large dataset with queries and qrels from `https://ir-datasets.com`

## Retrieval Models

The gateway and UI support:

- TF-IDF Vector Space Model
- BM25 with configurable `k1` and `b`
- Embedding search with Sentence Transformers and FAISS
- Hybrid Parallel Search with score fusion
- Hybrid Serial Search with BM25 candidate generation and semantic reranking

## Additional Features

The project includes three additional features for a 7-member group:

- Vector stores using FAISS.
- Personalization using recent search history.
- Documents/result clustering in the UI analytics tab.

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the demo services on Windows:

```bat
run_services.bat
```

Then open:

```text
http://localhost:8501
```

## Evaluation

Run the services first, then run:

```bash
python services/evaluation_service/evaluate_cranfield_compare.py
python services/evaluation_service/evaluate_scifact.py
```

The generated reports are saved in `reports/` and displayed in the UI Evaluation tab.

The evaluation scripts compare:

- TF-IDF
- BM25
- Semantic
- Hybrid Parallel
- Hybrid Serial

Metrics:

- Precision@10
- Recall@10
- MRR
- MAP
- nDCG@10

## Final Submission Checklist

- Replace development datasets with two approved datasets above 200K documents.
- Build indexes for both final datasets.
- Re-run evaluation for every retrieval model and every dataset.
- Add before/after evaluation for the three additional features.
- Include Arabic report, architecture diagram, dataset description, service description, evaluation analysis, and team task distribution.
