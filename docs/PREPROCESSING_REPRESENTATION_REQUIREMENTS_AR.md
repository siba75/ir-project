# مطابقة متطلبات معالجة البيانات وتمثيل الوثائق

## 1. Data Pre-Processing

تم تطبيق معالجة البيانات والاستعلامات في الخدمات التالية:

- `services/preprocessing_service/main.py`
- `services/refinement_service/main.py`
- `services/retrieval_service/main.py`
- `services/retrieval_service/build_ir_dataset_resources.py`

خطوات المعالجة المطبقة:

| الخطوة | مطبقة؟ | ملاحظات |
| --- | --- | --- |
| Normalization | نعم | lowercase، حذف الروابط، حذف الرموز، توحيد المسافات |
| Tokenization | نعم | تقسيم النص إلى tokens |
| Stopword Removal | نعم | خيار قابل للتفعيل من الواجهة |
| Stemming | نعم | باستخدام PorterStemmer، خيار من الواجهة |
| Lemmatization | نعم | باستخدام WordNetLemmatizer، خيار من الواجهة |

تمت إضافة Lemmatization كخيار مستقل حتى يدعم النظام stemming و lemmatization حسب الحاجة، مع إبقاء stemming غير مفعل افتراضياً لأن تقييم Quora الحالي كان أفضل بدون تغيير عدواني للكلمات.

## 2. Document and Query Representation

تم تطبيق كل طرق التمثيل المطلوبة على Quora:

| التمثيل المطلوب | التطبيق في المشروع |
| --- | --- |
| VSM TF-IDF | `dataset_tfidf_search` في Retrieval Service |
| Embedding | FAISS index مبني باستخدام SentenceTransformer، مع دعم LSA TF-IDF+SVD |
| BM25 | `BM25Okapi` وفهرس `indexes/quora/bm25.pkl` |
| Hybrid Parallel | دمج BM25 و Semantic scores بعد normalization |
| Hybrid Serial | BM25 candidate generation ثم Semantic FAISS re-ranking |

## ملاحظات المطلوب

### Hybrid Parallel Fusion

في `hybrid_search` يتم حساب:

- BM25 scores
- Semantic FAISS scores
- normalization لكل نوع score
- score نهائي باستخدام:

```text
final_score = bm25_weight * bm25_score + semantic_weight * semantic_score
```

الأوزان قابلة للتعديل من الواجهة عند اختيار Hybrid Parallel.

### BM25 Parameters

واجهة المستخدم تعرض معاملات BM25:

- `k1 = 1.5`
- `b = 0.75`

لكنها مقفلة في الواجهة لأن فهرس Quora الكامل مبني مسبقاً بهذه القيم لتقليل وقت الاسترجاع على جهاز 8GB RAM. تغيير القيم يتطلب إعادة بناء BM25 index، لذلك تم توثيق السبب في الواجهة والتقرير.

### Hybrid Serial / Parallel في الواجهة

واجهة Streamlit تحتوي خيارين مستقلين:

- `Hybrid Parallel`
- `Hybrid Serial`

في Hybrid Serial يمكن تغيير `Initial Candidates` من الواجهة، أما Hybrid Parallel فيمكن تغيير أوزان BM25 و Semantic.

## الخلاصة

البندان 1 و2 مطبقان عملياً وموثقان:

- preprocessing كامل مع stemming و lemmatization و normalization.
- كل تمثيلات الوثائق المطلوبة موجودة.
- hybrid parallel يستخدم fusion method.
- hybrid serial و parallel قابلان للاختيار من واجهة المستخدم.
- BM25 parameters موضحة ومبررة بسبب الفهرس المسبق.
