# قائمة جاهزية مشروع نظم استرجاع المعلومات

## المنجز حالياً

- بنية SOA مقسمة إلى خدمات مستقلة.
- اعتماد Dataset واحدة حسب موافقة المعيدة: `beir/quora/test`.
- Dataset تحقق الشرط المطلوب: أكثر من 200K وثيقة وتحتوي على queries و qrels.
- واجهة Streamlit لاختيار نموذج الاسترجاع وتنفيذ البحث على Quora.
- دعم TF-IDF و BM25 و Embedding و Hybrid Parallel و Hybrid Serial.
- التحكم بمعاملات BM25: `k1` و `b`.
- استخدام FAISS كـ vector store للبحث الدلالي.
- فهرس Quora المتجهي الحالي مبني بطريقة `sentence_transformer`، والكود يدعم أيضاً `lsa_tfidf_svd`.
- Query refinement بإضافة مرادفات.
- تقارير تقييم تعرض Precision@10 و Recall@10 و MRR و MAP و nDCG@10.
- ثلاث ميزات إضافية ظاهرة:
  - Vector stores باستخدام FAISS.
  - Personalization باستخدام سجل البحث.
  - Result clustering للوثائق المسترجعة في الواجهة.

## نقاط يجب إكمالها قبل التسليم النهائي

1. تشغيل الخدمات من `run_services.bat` والتأكد أن الواجهة تعمل على `http://localhost:8501`.
2. تشغيل تقييم Quora عند الحاجة:
   `python services/evaluation_service/evaluate_dataset_direct.py quora --max-queries 200`
3. تضمين نتائج التقرير الموجود في `reports/quora_evaluation_results.json` ضمن التقرير النهائي.
4. شرح أن التقييم تم على 200 query من qrels، وهو عدد أكبر ومناسب للتقرير مع مراعاة وقت التشغيل على جهاز 8GB RAM.
5. تشغيل أو عرض كل النماذج:
   - TF-IDF
   - BM25
   - Embedding
   - Hybrid Parallel
   - Hybrid Serial
6. توثيق الميزات الإضافية الثلاث ضمن التقرير وشرح أثرها.
7. كتابة تقرير عربي يتضمن:
   - وصف Dataset Quora.
   - وصف كل Service.
   - مخطط Architecture.
   - شرح SOA والتواصل بين الخدمات.
   - تحليل نتائج التقييم.
   - سبب اختيار معاملات BM25.
   - تقسيم العمل بين أعضاء الفريق السبعة.

## ملاحظة مهمة

النظام الحالي مبني على `beir/quora/test` فقط حسب موافقة المعيدة على Dataset واحدة، وهي تحقق شرط الحجم ووجود queries/qrels.

- ملخص التقييم قبل/بعد الميزات موجود في: docs/EVALUATION_SUMMARY_AR.md


