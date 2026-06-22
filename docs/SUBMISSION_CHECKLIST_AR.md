# Checklist التسليم والمناقشة

## Dataset

- الداتا الوحيدة المستخدمة: `beir/quora/test`.
- لا يوجد تقسيم للداتا.
- التقييم يستخدم كل queries الموجودة في qrels:
  - `total_qrels_queries = 10000`
  - `evaluated_queries = 10000`
  - `evaluation_scope = all_qrels_queries`

## Storage and Caching

- الوثائق الأصلية مخزنة في SQLite:
  - `indexes/quora/documents.sqlite`
  - جدول `documents(doc_id, content)`
  - عدد الوثائق: `522931`
- الواجهة تعرض `doc_id` والنص الأصلي الكامل للدوكيومنت.
- الفهارس والموديلات محفوظة كموارد مضغوطة:
  - `indexes/quora/bm25.pkl.gz`
  - `indexes/quora/tfidf.pkl.gz`
  - `indexes/quora/faiss.index.gz`
  - `indexes/quora/metadata.pkl.gz`
  - `indexes/quora/inverted_index.json.gz`
- ملف توصيف الموارد:
  - `indexes/quora/resource_manifest.json`

## Libraries

- BM25 مستخدم من مكتبة:
  - `rank_bm25.BM25Okapi`
- TF-IDF مستخدم من مكتبة:
  - `sklearn.feature_extraction.text.TfidfVectorizer`
- Semantic/vector store:
  - `FAISS`

## Evaluation

- التقرير النهائي:
  - `reports/quora_evaluation_results.json`
- يحتوي قبل وبعد الميزات الإضافية.
- يحتوي المقاييس:
  - `Precision@10`
  - `Recall@10`
  - `MAP`
  - `nDCG@10`
  - `MRR`
- الواجهة تعرض جداول ورسومات قبل/بعد، مع تركيز إضافي على `MAP` و `nDCG@10`.

## UI

- اختيار موديل البحث من الواجهة:
  - TF-IDF
  - BM25
  - Semantic
  - Hybrid Parallel
  - Hybrid Serial
- التحكم بمعاملات BM25:
  - `k1`
  - `b`
- التحكم بأوزان Hybrid Parallel:
  - `BM25 Weight`
  - `Semantic Weight`
- عرض charts للـ:
  - score comparison
  - ranking curve
  - clustering
  - evaluation before/after

## SOA

- الخدمات منفصلة:
  - preprocessing service
  - indexing service
  - retrieval service
  - ranking/evaluation service
  - refinement service
  - gateway service
  - Streamlit UI

