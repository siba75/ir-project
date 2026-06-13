# مطابقة متطلبات الفهرسة ومعالجة الاستعلام والترتيب

## 3. Indexing

تم تطبيق الفهرسة على مستويين:

- `services/indexing_service/main.py` يبني Inverted Index لأي مجموعة وثائق ترسل إلى الخدمة، مع حفظ postings و document lengths وإحصائيات الفهرس.
- `services/retrieval_service/build_ir_dataset_resources.py` يبني موارد Quora الفعلية:
  - `indexes/quora/inverted_index.json`
  - `indexes/quora/bm25.pkl`
  - `indexes/quora/faiss.index`
  - `indexes/quora/metadata.pkl`

الفهرسة تستخدم نفس أسلوب التنظيف الأساسي: lowercasing، حذف الروابط، حذف الرموز غير المهمة، وتوحيد المسافات. هذا يجعل مصطلحات الفهرسة متوافقة مع مصطلحات البحث.

## 4. Query Processing

معالجة الاستعلامات مطبقة في:

- `services/preprocessing_service/main.py`
- `services/refinement_service/main.py`
- `services/retrieval_service/main.py`

الخطوات الأساسية:

- تحويل النص إلى lowercase.
- حذف URLs.
- حذف الرموز غير الأبجدية والرقمية.
- توحيد المسافات.
- tokenization.
- حذف stopwords اختيارياً.
- stemming اختيارياً.

بهذا تكون الاستعلامات ممثلة بنفس طريقة تمثيل الوثائق داخل الفهارس النصية.

## 5. Query Refinement

تم تطبيق Query Refinement في `services/refinement_service/main.py` من خلال:

- تنظيف الاستعلام.
- حذف stopwords حسب اختيار المستخدم.
- stemming اختياري.
- query expansion بإضافة مرادفات مناسبة لطبيعة Quora مثل:
  - `programming -> coding, development`
  - `job -> career, work`
  - `phone -> mobile, smartphone`
  - `health -> medical, wellness`

كما تم تقييم أثر Query Expansion رقمياً في تقرير قبل/بعد الميزات الإضافية.

## 6. Query Matching & Ranking

مطابقة الاستعلام وترتيب النتائج مطبقان في `services/retrieval_service/main.py`:

| النموذج | طريقة المطابقة والترتيب |
| --- | --- |
| TF-IDF | حساب وزن term frequency مع IDF ثم ترتيب تنازلي |
| BM25 | استخدام BM25Okapi وترتيب الوثائق حسب score |
| Semantic FAISS | تحويل الاستعلام إلى embedding والبحث في FAISS حسب inner product/cosine similarity بعد normalization |
| Hybrid Parallel | دمج BM25 score و Semantic score بعد normalization |
| Hybrid Serial | اختيار candidates عبر BM25 ثم semantic reranking |

كل endpoints ترجع:

- `rank`
- `doc_id`
- `score`
- `text`

وهذا يثبت أن النتائج لا تُسترجع فقط، بل يتم ترتيبها حسب درجة المطابقة.

## الخلاصة

البنود 3 و4 و5 و6 مطبقة ومترابطة:

- الوثائق تفهرس بعد preprocessing.
- الاستعلامات تنظف وتمثل بنفس منطق الفهرسة.
- refinement يضيف تحسينات اختيارية قابلة للتقييم.
- ranking يتم حسب score مناسب لكل نموذج.
