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
  - يستخدم analyzer موحد من المشروع: `text_processing.preprocess_text`
  - تم إيقاف الاعتماد على preprocessing الافتراضي للمكتبة عبر تمرير callable analyzer.
- Semantic/vector store:
  - `FAISS`

## Data Processing

- تجهيز البيانات يتم محلياً قبل التشغيل.
- الوثائق تقرأ من الداتا وتتحول إلى نص موحد عبر `doc_to_text`.
- إذا كان الدوكيومنت فيه metadata مثل `title`, `text`, `body`, `abstract`, `author` يتم دمج الحقول الموجودة قبل الفهرسة.
- preprocessing موحد لكل:
  - BM25
  - TF-IDF
  - query matching
- SQLite document loading يستخدم batch insert بحجم `10000`.

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
- تجربة الميزات الإضافية بشكل مستقل:
  - `Query Expansion` تشغيل/إيقاف.
  - `Personalization` تشغيل/إيقاف.
  - `Result Clustering` تشغيل/إيقاف.
- عرض charts للـ:
  - score comparison
  - ranking curve
  - clustering
  - evaluation before/after

## Personalization and Query Suggestion

- عند تفعيل `Personalization` من الواجهة، النظام يستخدم سجل البحث السابق كـ IR mini-system:
  - يأخذ آخر queries من history.
  - يحول history والـ query الحالية إلى TF-IDF vectors.
  - يحسب similarity بين query الحالية وqueries السابقة.
  - يبني `interest vector` للمستخدم بوزن similarity + recency.
  - يدمج `query vector` مع `interest vector`:
    - `query_vector_weight = 0.70`
    - `interest_vector_weight = 0.30`
  - يستخرج terms من الـ combined vector ويضيفها خلف الكواليس للـ query.
  - يولد query suggestions مبنية على تشابه history وليس بحث نصي بسيط.
- الواجهة تعرض:
  - User Interest Terms
  - Terms Selected After Query-Interest Fusion
  - Most Similar History Queries
  - IR-based Query Suggestions
  - Charts للأوزان والتشابه.

## SOA

- الخدمات منفصلة:
  - preprocessing service
  - indexing service
  - retrieval service
  - ranking/evaluation service
  - refinement service
  - gateway service
  - Streamlit UI
