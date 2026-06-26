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

ملف تجميع صغير:
- يعرّف الأسماء التي يستدعيها `main.py`.
- يوجه التنفيذ إلى ملفات البحث المتخصصة.
- يحافظ على API داخلي ثابت بعد تقسيم الكود.

### `services/retrieval_service/tfidf_retrieval.py`

بحث TF-IDF الحقيقي على داتا Quora:
- يستخدم `sklearn TfidfVectorizer`.
- يستعمل matrix محفوظة ومضغوطة.
- يرجع النص الأصلي من SQLite حسب `doc_id`.

### `services/retrieval_service/bm25_retrieval.py`

بحث BM25:
- يستخدم مكتبة `rank_bm25`.
- يطبق `k1` و `b` وقت تنفيذ الاستعلام من الواجهة.
- يعرض processed query والنتائج المرتبة.

### `services/retrieval_service/semantic_retrieval.py`

البحث الدلالي:
- FAISS vector index.
- LSA TF-IDF SVD أو Sentence Transformer حسب metadata.
- مسؤول عن embedding query والبحث داخل الفهرس.

### `services/retrieval_service/hybrid_retrieval.py`

البحث الهجين:
- Hybrid Parallel: دمج scores بين BM25 و Semantic.
- Hybrid Serial: BM25 candidates ثم Semantic re-ranking.
- يحافظ على scores منفصلة لشرح الفرق أمام المعيدة.

### `services/retrieval_service/lexical_retrieval.py`

منطق lexical مشترك:
- tokenization للاستعلام.
- حساب BM25 scores.

### `services/retrieval_service/retrieval_common.py`

دوال مشتركة بين خوارزميات الاسترجاع:
- validations.
- score normalization.
- قراءة النص الأصلي من قاعدة SQLite.
- بناء ranked results.

### `services/retrieval_service/text_processing.py`

المصدر الموحد لمعالجة النصوص:
- `preprocess_text`: نفس التنظيف والتوكنة المستخدمة في BM25 وTF-IDF والاستعلامات.
- `doc_to_text`: يجمع حقول الدوكيومنت مثل `title`, `text`, `body`, `abstract` إذا كانت موجودة.
- `iter_batches`: تقسيم البيانات إلى batches أثناء تجهيز الموارد.

### `services/retrieval_service/vector_builders.py`

بناء vector resources:
- LSA TF-IDF SVD vectors.
- SentenceTransformer vectors.
- FAISS index.
- metadata الخاصة بالتمثيل الدلالي.

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
- يبني TF-IDF باستخدام analyzer مشترك: `text_processing.preprocess_text`.
- يدخل الوثائق في SQLite على batches.
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

نقطة تشغيل Streamlit فقط:
- page config.
- تهيئة session state.
- ربط sidebar وsearch وtabs.

### `ui/streamlit_app/ui_sidebar.py`

إعدادات الواجهة:
- اختيار retrieval mode.
- BM25 parameters.
- hybrid weights.
- تفعيل الميزات الإضافية مثل personalization وclustering.
- عرض حالة الخدمات.

### `ui/streamlit_app/ui_search.py`

تشغيل البحث من الواجهة:
- بناء payload للـ Gateway.
- إرسال request.
- حفظ آخر نتيجة وsearch history.

### `ui/streamlit_app/ui_results.py`

عرض النتائج:
- الوثيقة الأصلية كاملة من SQLite.
- score النهائي.
- BM25/Semantic scores حسب المود.

### `ui/streamlit_app/ui_analytics.py`

التحليلات والميزات الإضافية:
- score charts.
- BM25 parameter comparison.
- result clustering.
- personalization profile.

### `ui/streamlit_app/ui_evaluation.py`

تبويب التقييم:
- before/after tables.
- Precision@10 وRecall وMRR وMAP وnDCG@10.
- charts للتركيز على MAP وnDCG.

### `ui/streamlit_app/ui_pipeline_export.py`

تبويب pipeline وexport:
- يوضح خطوات التنفيذ.
- يعرض resource manifest.
- يصدّر النتائج CSV/JSON.

### `ui/streamlit_app/ui_sections.py`

ملف تجميع صغير للـ tabs:
- يربط نتائج البحث والتحليلات والتقييم والـ pipeline والـ export.

### `ui/streamlit_app/ui_helpers.py`

دوال مساعدة للواجهة:
- تحميل نتائج التقييم.
- بناء الجداول.
- clustering.
- highlighting.
- charts helpers.

### `ui/streamlit_app/ui_theme.py`

تنسيق الواجهة:
- light/dark colors.
- CSS الخاص بالكروت، النتائج، التابات، والـ metrics.

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
