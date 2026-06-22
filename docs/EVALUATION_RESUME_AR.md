# متابعة التقييم من مكان التوقف

## كيف يعمل

سكربت التقييم يحفظ checkpoints داخل:

```text
reports/evaluation_checkpoints
```

لكل مرحلة ونموذج، مثل:

- قبل الميزات / BM25
- قبل الميزات / Semantic
- بعد الميزات / Hybrid Parallel

إذا توقف التقييم، لا تعيدي من الصفر. شغلي نفس الأمر مرة ثانية:

```bash
python services/evaluation_service/evaluate_dataset_direct.py quora
```

أو من النوتبوك شغلي خلية:

```text
Start Full Evaluation
```

وسيكمل من آخر checkpoint محفوظ.

## كيف أعرف أنه عم يكمل؟

في ملف log:

```text
reports/quora_full_evaluation.log
```

وفي بداية كل model سيظهر سطر مثل:

```text
Resuming quora/before_features/bm25: 4300/10000 queries already done
```

هذا يعني أنه لم يبدأ من الصفر.

## إذا أردت البدء من الصفر

استخدمي:

```bash
python services/evaluation_service/evaluate_dataset_direct.py quora --fresh
```

أو احذفي مجلد:

```text
reports/evaluation_checkpoints
```

ثم شغلي التقييم من جديد.
