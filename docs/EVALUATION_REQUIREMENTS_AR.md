# مطابقة متطلبات تقييم النظام

## Dataset المستخدمة

التطبيق النهائي يستخدم Dataset واحدة حقيقية:

- Dataset: `beir/quora/test`
- الاسم الداخلي: `quora`
- عدد الوثائق: 522,931
- عدد qrels queries: 10,000
- عدد qrel judgments: 15,675

كل نماذج الاسترجاع تبحث داخل كامل فهارس Quora، وليس داخل بيانات صغيرة للتجريب.

## المقاييس المطلوبة

| المقياس | حالة التطبيق |
| --- | --- |
| MAP | مطبق |
| Recall | مطبق كـ Recall@10 |
| Precision@10 | مطبق |
| nDCG | مطبق كـ nDCG@10 |

## النماذج المقيمة

يحسب سكربت التقييم النتائج لكل النماذج المطلوبة:

- TF-IDF
- BM25
- Semantic FAISS
- Hybrid Parallel
- Hybrid Serial

## قبل وبعد الميزات الإضافية

ملف التقييم ينتج:

- `before_features`: قبل تطبيق Query Expansion.
- `after_features`: بعد تطبيق Query Expansion.
- `comparison`: فرق الأداء بين قبل وبعد.

## التقييم المطلوب للمقابلة

حسب طلب المعيدة، التقييم النهائي يجب أن يستخدم كل queries الموجودة في qrels. لذلك التشغيل النهائي يكون بدون `--max-queries` أو باستخدام `--all-queries`:

```bash
python services/evaluation_service/evaluate_dataset_direct.py quora
```

أو:

```bash
python services/evaluation_service/evaluate_dataset_direct.py quora --all-queries
```

عند التشغيل يطبع السكربت:

- عدد queries الكلي في qrels.
- عدد qrel judgments الكلي.
- عدد queries المختارة للتقييم.
- scope التقييم.

لـ Quora يجب أن يظهر:

```text
Total qrels queries: 10000
Total qrel judgments: 15675
Queries selected for evaluation: 10000
Evaluation scope: all_qrels_queries
```

## التقرير الحالي

التقرير الحالي محفوظ في:

```text
reports/quora_evaluation_results.json
```

وهو مضبوط ليعرض:

- عدد الاستعلامات المقيمة.
- `top_k`.
- نتائج قبل الميزات.
- نتائج بعد الميزات.
- فرق الأداء.

## تشغيل تقييم سريع للتجريب فقط

```bash
python services/evaluation_service/evaluate_dataset_direct.py quora --max-queries 200
```

هذا للتجريب فقط وليس التقرير النهائي المطلوب للمقابلة، لأن المعيدة طلبت استخدام كل qrels queries.

## تشغيل التقييم على كل qrels

التشغيل النهائي لكل qrels:

```bash
python services/evaluation_service/evaluate_dataset_direct.py quora
```

هذا يشغل التقييم على كل 10,000 qrels queries، وقد يستغرق ساعات طويلة لأن التقييم ينفذ خمسة نماذج قبل وبعد الميزات الإضافية.

## الخلاصة

متطلبات التقييم مطبقة. الاسترجاع يتم على كل وثائق Quora، وسكربت التقييم يدعم عينة محددة أو كل qrels queries حسب وقت التشغيل المتاح.
