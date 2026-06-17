# تدقيق الاعتماد على Dataset الحقيقية

## النتيجة

التطبيق النهائي يعتمد على Dataset واحدة حقيقية فقط:

- `beir/quora/test`
- الاسم الداخلي: `quora`

## المسارات الفعلية في التطبيق

| الجزء | المسار المستخدم |
| --- | --- |
| TF-IDF | `/search/dataset/tfidf` |
| BM25 | `/search/dataset/bm25` |
| Semantic FAISS | `/search/semantic` |
| Hybrid Parallel | `/search/hybrid` |
| Hybrid Serial | `/search/hybrid/serial` |
| Gateway | `/search/full` |
| Indexing stats | `/index/stats` من فهرس Quora الجاهز |
| Index term lookup | `/index/term/{term}` من فهرس Quora الجاهز |
| Evaluation | `evaluate_dataset_direct.py` باستخدام qrels الحقيقية |

## الأشياء التجريبية التي تمت إزالتها

- البحث على documents صغيرة مرسلة داخل request.
- `tfidf_search()` التجريبية.
- endpoints القديمة مثل `/search` و `/search/bm25`.
- بناء index مؤقت من documents صغيرة داخل `Indexing Service`.
- اختيار Dataset من الواجهة.

## الموارد الحقيقية المستخدمة

- `datasets/quora/queries.json`
- `datasets/quora/qrels.json`
- `indexes/quora/inverted_index.json`
- `indexes/quora/bm25.pkl`
- `indexes/quora/faiss.index`
- `indexes/quora/metadata.pkl`
- `reports/quora_evaluation_results.json`

## التقييم

الاسترجاع يتم دائماً على كامل وثائق Quora. أما التقييم فيدعم وضعين:

- تقييم سريع على عدد محدد من qrels queries مثل 200.
- تقييم كامل على كل 10,000 qrels queries باستخدام:

```bash
python services/evaluation_service/evaluate_dataset_direct.py quora
```

كما يوجد نوتبوك جاهز للمقابلة يطبع عدد qrels queries ويشغل التقييم الكامل:

```text
notebooks/quora_full_evaluation.ipynb
```

## الخلاصة

الواجهة، Gateway، Retrieval، Indexing، والتقييم كلها مبنية على Quora الحقيقية، ولا يوجد مسار مستخدم في التطبيق النهائي يعتمد على بيانات صغيرة للتجريب.
