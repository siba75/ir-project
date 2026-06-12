# تقرير مشروع نظم استرجاع المعلومات 2026

## عنوان المشروع

بناء نظام استرجاع معلومات متعدد الخدمات يدعم البحث النصي، البحث الدلالي، والبحث الهجين.

## وصف عام

يهدف المشروع إلى بناء محرك بحث قادر على استقبال استعلام المستخدم، معالجته، تطبيق تحسينات عليه، ثم استرجاع وترتيب الوثائق الأكثر صلة من مجموعتي بيانات. يعتمد النظام على بنية Service Oriented Architecture بحيث تكون كل مهمة ضمن خدمة مستقلة قابلة للتشغيل والتطوير والاختبار بشكل منفصل.

## بنية النظام

الخدمات الأساسية:

- Preprocessing Service: تنظيف النصوص، normalization، حذف stopwords، stemming.
- Indexing Service: بناء inverted index وإرجاع إحصائيات الفهرسة.
- Retrieval Service: تنفيذ TF-IDF و BM25 و Embedding و Hybrid Retrieval.
- Query Refinement Service: تحسين الاستعلام وإضافة مرادفات.
- Evaluation Service: حساب مقاييس تقييم نظم استرجاع المعلومات.
- Gateway Service: تنسيق الطلب بين تحسين الاستعلام وخدمة الاسترجاع.
- Streamlit UI: واجهة المستخدم.

آلية التواصل بين الخدمات تتم باستخدام REST API.

## نماذج تمثيل الوثائق والاستعلامات

- TF-IDF Vector Space Model.
- BM25 مع التحكم بالمعاملات `k1` و `b`.
- Embedding باستخدام Sentence Transformers.
- Hybrid Parallel عن طريق دمج درجات BM25 و Embedding.
- Hybrid Serial عن طريق BM25 candidate generation ثم semantic reranking.

## الميزات الإضافية

لأن عدد أعضاء المجموعة 7، تم تضمين ثلاث ميزات إضافية:

- Vector stores باستخدام FAISS.
- Personalization بالاعتماد على آخر استعلامات المستخدم.
- Result clustering لتجميع النتائج المسترجعة في الواجهة.

## التقييم

يجب تشغيل التقييم لكل Dataset ولكل نموذج:

- TF-IDF
- BM25
- Semantic
- Hybrid Parallel
- Hybrid Serial

المقاييس المستخدمة:

- Precision@10
- Recall@10
- MRR
- MAP
- nDCG@10

يجب إضافة جدول مقارنة قبل وبعد الميزات الإضافية، وتحليل أثر كل ميزة على جودة النتائج وسرعة الاسترجاع.

## ملاحظة حول مجموعات البيانات

النسخة الحالية تستخدم Cranfield و SciFact لأغراض التطوير والعرض. حسب نص المشروع النهائي يجب استخدام مجموعتي بيانات تحتوي كل واحدة منهما على أكثر من 200K وثيقة وأن تحتوي على queries و qrels. لذلك يجب استبدال مجموعات التطوير بمجموعات كبيرة معتمدة قبل التسليم النهائي.

## تقسيم العمل

يضاف هنا أسماء أعضاء الفريق السبعة وتوزيع المهام:

| العضو | المهمة |
| --- | --- |
| الطالب 1 | Preprocessing Service |
| الطالب 2 | Indexing Service |
| الطالب 3 | Retrieval Service |
| الطالب 4 | Query Refinement |
| الطالب 5 | Evaluation |
| الطالب 6 | UI + Gateway |
| الطالب 7 | Report + Experiments + Additional Features |
