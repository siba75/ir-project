# بنية الكود للمناقشة

## Services

### `services/preprocessing_service/main.py`

مسؤول عن تنظيف النصوص:
- lowercase
- إزالة الرموز والروابط
- stopwords
- stemming
- lemmatization

### `services/refinement_service/main.py`

مسؤول عن تحسين الاستعلام قبل البحث:
- تطبيق preprocessing على query.
- query expansion.
- إرجاع refined query.

### `services/indexing_service/main.py`

مسؤول عن فحص الفهرس الجاهز:
- قراءة `inverted_index.json.gz`.
- عرض إحصائيات الفهرس.
- عرض postings لأي term.

### `services/retrieval_service/main.py`

واجهة API فقط:
- يستقبل requests.
- يستدعي دوال البحث من `retrieval_core.py`.
- يرجع النتائج.

### `services/retrieval_service/retrieval_core.py`

منطق البحث والترتيب:
- TF-IDF باستخدام `sklearn TfidfVectorizer`.
- BM25 باستخدام `rank_bm25`.
- Semantic Search باستخدام FAISS.
- Hybrid Parallel.
- Hybrid Serial.
- قراءة النص الأصلي من SQLite حسب `doc_id`.

### `services/retrieval_service/dataset_manager.py`

تحميل موارد الداتا:
- BM25 compressed resource.
- TF-IDF compressed resource.
- FAISS index.
- metadata.
- SQLite document store.
- caching للموارد.

### `services/retrieval_service/prepare_submission_resources.py`

سكريبت تجهيز موارد التسليم:
- يبني `documents.sqlite`.
- يضغط الفهارس والموديلات.
- يبني TF-IDF resource.
- يكتب `resource_manifest.json`.

### `services/gateway_service/main.py`

واجهة النظام المركزية:
- يستقبل طلب البحث من الواجهة.
- يتحقق من parameters.
- يستدعي refinement service.
- يطبق personalization إذا مفعلة.
- يختار retrieval strategy.
- يرجع response موحد للواجهة.

### `services/gateway_service/retrieval_strategies.py`

يحدد كيف كل retrieval mode يتحول إلى request:
- TF-IDF
- BM25
- Semantic
- Hybrid Parallel
- Hybrid Serial

### `services/gateway_service/personalization.py`

طلب الـ personalization:
- يبني user-interest vector من search history.
- يحسب similarity بين query الحالية والـ history.
- يدمج query vector مع interest vector.
- يولد query suggestions مبنية على IR.

### `services/evaluation_service/evaluate_dataset_direct.py`

تشغيل التقييم النهائي:
- يستخدم كل qrels queries.
- يحسب before/after.
- يحسب MAP و nDCG و Precision و Recall و MRR.
- يدعم checkpoint/resume.

## UI

### `ui/streamlit_app/app.py`

واجهة Streamlit الرئيسية:
- sidebar controls.
- search.
- tabs.
- عرض النتائج.
- عرض evaluation.
- عرض pipeline.

### `ui/streamlit_app/ui_helpers.py`

دوال مساعدة للواجهة:
- تحميل نتائج التقييم.
- بناء الجداول.
- clustering.
- highlighting.
- charts helpers.
- theme.

## Data and Models

### `datasets/quora`

- `queries.json`
- `qrels.json`

### `indexes/quora`

- `documents.sqlite`: النصوص الأصلية.
- `bm25.pkl.gz`: BM25 model.
- `tfidf.pkl.gz`: TF-IDF model.
- `faiss.index.gz`: نسخة مضغوطة للتسليم.
- `faiss.index`: نسخة جاهزة للتحميل السريع.
- `metadata.pkl.gz`: metadata.
- `inverted_index.json.gz`: inverted index.
- `resource_manifest.json`: وصف موارد التسليم.

### `reports`

- `quora_evaluation_results.json`: نتائج التقييم النهائي.
- `evaluation_checkpoints`: checkpoints للتقييم الكامل.

