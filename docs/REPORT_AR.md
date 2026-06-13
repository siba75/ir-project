# تقرير مشروع نظم استرجاع المعلومات 2026

## عنوان المشروع

بناء نظام استرجاع معلومات متعدد الخدمات يدعم البحث النصي، البحث الدلالي، والبحث الهجين.

## وصف عام

يهدف المشروع إلى بناء محرك بحث قادر على استقبال استعلام المستخدم، معالجته، تطبيق تحسينات عليه، ثم استرجاع وترتيب الوثائق الأكثر صلة من Dataset واحدة مستوفية للشروط حسب موافقة المعيدة. يعتمد النظام على بنية Service Oriented Architecture بحيث تكون كل مهمة ضمن خدمة مستقلة قابلة للتشغيل والتطوير والاختبار بشكل منفصل.

## مجموعة البيانات

تم اعتماد Dataset:

- الاسم ضمن `ir-datasets`: `beir/quora/test`
- المجال: أسئلة Quora المتشابهة.
- عدد الوثائق: 522,931 وثيقة.
- عدد الاستعلامات: 10,000 استعلام.
- تحتوي على qrels للاستعلامات، لذلك يمكن استخدامها في تقييم نماذج الاسترجاع.

تم اختيار هذه Dataset لأنها تحقق شرط أكثر من 200K وثيقة، وتحتوي على docs و queries و qrels، كما أنها مناسبة لجهاز بذاكرة 8GB مقارنة بمجموعات أكبر بكثير مثل MS MARCO.

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

تم توثيق البنية المعمارية و Design Patterns المستخدمة بالتفصيل في:

`docs/SOA_DESIGN_PATTERNS_AR.md`

يعتمد المشروع على عدة patterns واضحة:

- API Gateway لتوحيد نقطة الدخول من الواجهة إلى الخدمات.
- Strategy Pattern لاختيار نموذج الاسترجاع المناسب داخل Gateway.
- Factory Method لبناء payload مختلف لكل نموذج بحث.
- Repository / Resource Manager لعزل تحميل Dataset والفهارس.
- Singleton Cache لتقليل إعادة تحميل موارد Quora ونموذج SentenceTransformer.
- Pipeline Pattern لتمرير الاستعلام عبر refinement ثم retrieval ثم ranking.

كما تم توثيق مطابقة متطلبات الفهرسة ومعالجة الاستعلام وتحسينه والترتيب في:

`docs/IR_CORE_REQUIREMENTS_AR.md`

كما تم توثيق مطابقة متطلبات preprocessing وتمثيل الوثائق في:

`docs/PREPROCESSING_REPRESENTATION_REQUIREMENTS_AR.md`

## نماذج تمثيل الوثائق والاستعلامات

- TF-IDF Vector Space Model.
- BM25 مع التحكم بالمعاملات `k1` و `b`.
- Embedding باستخدام FAISS. يدعم النظام طريقتين لبناء الفهرس المتجهي: `sentence_transformer` و `lsa_tfidf_svd`. الفهرس الحالي الخاص بـ Quora مبني باستخدام SentenceTransformer.
- Hybrid Parallel عن طريق دمج درجات BM25 و Embedding.
- Hybrid Serial عن طريق BM25 candidate generation ثم semantic reranking.

تم بناء Inverted Index و BM25 index و FAISS vector index لمجموعة Quora، وتتم معالجة الاستعلامات بنفس خطوات التنظيف الأساسية المستخدمة أثناء الفهرسة لضمان التوافق بين تمثيل الوثائق وتمثيل الاستعلام. يدعم النظام Normalization و Stopword Removal و Stemming و Lemmatization كخيارات preprocessing.

بالنسبة لمعاملات BM25، يتم عرض `k1` و `b` في الواجهة عند استخدام BM25 أو النماذج الهجينة، ويمكن تغييرها لكل query. يطبق النظام القيم المختارة أثناء حساب BM25 scores، وتعرض الواجهة جدول مقارنة مع التشغيل السابق لنفس الاستعلام لتوضيح أثر تغيير المعاملات على الترتيب والدرجات.

كما توفر الواجهة خيارين منفصلين للتمثيل الهجين: Hybrid Parallel و Hybrid Serial. في Hybrid Parallel يمكن تغيير أوزان BM25 و Semantic، وفي Hybrid Serial يمكن تغيير عدد الوثائق المرشحة الأولية قبل إعادة الترتيب الدلالي.

## الميزات الإضافية

لأن عدد أعضاء المجموعة 7، تم تضمين ثلاث ميزات إضافية:

- Vector stores باستخدام FAISS.
- Personalization بالاعتماد على آخر استعلامات المستخدم.
- Result clustering لتجميع النتائج المسترجعة في الواجهة.

## التقييم

يتم تشغيل التقييم على Dataset Quora لكل نموذج:
تم تشغيل التقييم على 200 استعلام من qrels، وتم توليد تقرير JSON في:

`reports/quora_evaluation_results.json`

كما تم تجهيز ملخص الجداول والتحليل في:

`docs/EVALUATION_SUMMARY_AR.md`

كما تم تجهيز ملف يطابق متطلبات التقييم بنداً بنداً في:

`docs/EVALUATION_REQUIREMENTS_AR.md`

يعرض التقرير ثلاث جداول: قبل الميزات الإضافية، بعد الميزات الإضافية، وفرق الأداء بين المرحلتين. أظهرت النتائج أن Semantic FAISS هو الأفضل على عينة التقييم الحالية من Quora، حيث حقق أعلى قيم في Recall@10 و MRR و MAP و nDCG@10. كما ظهر أن query expansion اليدوي لم يحسن الأداء على هذه العينة، لذلك بقيت هذه الميزة اختيارية في الواجهة.

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

تمت إضافة مقارنة قبل وبعد الميزات الإضافية، مع تحليل أثرها على جودة الاسترجاع في ملف ملخص التقييم.

## ملاحظة حول الاعتماد على Dataset واحدة

حسب موافقة المعيدة، تم تنفيذ المشروع على Dataset واحدة فقط بشرط أن تحقق متطلبات الحجم ووجود qrels. لذلك تم اعتماد `beir/quora/test`، وهي تحقق هذه الشروط.

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


