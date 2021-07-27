---  
layout: post
title: قاعدة بيانات مقالات جريدة الرياض عن كوفيد 19
icon: 📰
---  

يسعدني أنا والدكتورة [نجوى الغامدي][najwa] أن نشاركم قاعدة بيانات للإستخدام في الاغراض البحثية تتضمن المقالات المنشورة في جريدة الرياض السعودية والتي تشمل على كلمات لها علاقة في كوفيد-19 منذ البداية حتى 1 فبراير 2021. ومن الخطط ان يتم تحديث هذة البيانات بين فترة وأخرى.

## عن البيانات

البيانات هي جزء من بحث علمي تم تقديمة لعدد من المؤتمرات. تحتوي على جميع المقالات من صحيفة [الرياض][riyadh] التي كُتبت وفيها الكلمات التالية:

- كورونا
- كوفيد-19
- كوفيد المستجد
- حظر التجول
- منع التجول

## الهدف من البيانات

نهدف في هذة البيانات إفادة المحللين ومن يرغب بدراسة اللغة العربية واستخدامها في المقالات الصحفية.

## فائدة البيانات

- تعتبر البيانات هي أول واكبر قاعدة بيانات لمقالات صحفية وأخبار محلية أو دولية لها علاقة بكوفيد-19.
- يمكن الإستفادة من البيانات لرسم مخططات زمنية ودراسة السلوك.
- يمكن الإستفادة منها بتدريب وضبط نماذج اللغات في التعلم العميق.

## مصدر البيانات

تم جمع البيانات من الموقع الرسمي لجريدة الرياض السعودية بعد الموافقة منهم.

## حجم البيانات

كل كلمة يختلف عدد المقالات فيها:

| الكلمة | عدد المقالات |
| :--: | :------------------: |
| كورونا | 21961 |
| كوفيد-19 | 1266 |
| كوفيد المستجد | 3044 |
| حظر التجول | 1087 |
| منع التجول | 1255 |

العدد الأجمالي للمقالات بعد حذف المقالات المتكررة في حال كانت المقالة تحتوي على أكثر من كلمة من التي في الجدول السابق هو: **24084 سطر**

## قاموس البيانات

الجدول التالي يشرح كل عامود، نوع البيانات فيه ومثال عليها:

| Column | Description | Datatype | Example |
| :--: | :--: | :--: | :--: |
| ID * | الرقم المميز للمقالة | رقم | `1867288` |
| Category | في أي نوع من المقالات تم إضافة هذا المقال | نص | `مقالات اليوم` / `أخبار المناطق` |
| Source | مصدر الخبر، ممكن ان يكون مصدر رسمي للأخبار او كاتب المقالة | نص | `الرياض - واس` or `خالد بن علي المطرفي` |
| Date | التاريخ | تاريخ | `2020-03-27` |
| Time | وقت نشر المقالة، في حال كان بدون نص او Null يعني أنة لم يتم إضافتة | نص | `12:05:51` |
| Title | عنوان المقالة | نص | `أمير الجوف يشدّد على تطبيق الإجراءات الاحترازيه` |
| Subtitle | العنوان الفرعي للمقالة، لا يوجد في جمع المقالات | نص | `رأس اجتماع غرفة العمليات المشتركة` |
| Text | نص المقالة | نص | `شدّد صاحب السمو الملكي الأمير فيصل بن نواف بن عبدالعزيز...` |
| Image * | اذا كانت المقالة تحتوي على صورة واحدة، سيكون رابطها هنا | نص | `/media/thumb/af/1d/1000_5b1f4e7dc6.jpg` |
| Caption |اذا كانت المقالة تحتوي على صورة واحدة، سيكون عنوانها هنا | نص | `اليابان تخزن المزيد من النفط السعودي محققة انتعاشاً للطلب` |
| Images * | اذا كانت المقالة تحتوي على أكثر من صورة، ستكون روابطها هنا | مصفوفة | `['/media/thumb/08/d7/1000_a4964ed6ad.jpg', '/media/thumb/04/14/1000_127ad97cda.jpg']` | 
| Captions | اذا كانتا لمقالة تحتوي على أكثر من صورة، ستكون عناوينها هنا | مصفوفة | `['مواعيد إلكترونية لاستقبال المراجعين', 'تطبيق مواعيد الدخول في المحاكم']` |
| URL | رابط المقالة | نص | `http://www.alriyadh.com/1867288` |
| Terms | المصطلحات التي تم إستخدامها في المقالة من المصطلحات الخمسة | مصفوفة | `['كورونا', 'حظر التجول']` |
| FullText | تجميع للأعمدة الثلاثة Title و Subtitle و Text | نص | `أمير الجوف يشدّد على تطبيق الإجراءات الاحترازية\\n ورصد المخالفات رأس اجتماع ... في تحقيقها ولله الحمد` |
| FullTextCleaned | العامود FullText بعد تنظيفة ( حذف النصوص الأنجليزية، الأرقام، ورموز الانتقال لسطر جديد) | نص | `أمير الجوف يشدّد على تطبيق الإجراءات الاحترازية ورصد المخالفات رأس اجتماع ... في تحقيقها ولله الحمد` |
| FullTextWords | العامود FullTextCleaned بعد تقسيم الكلمات فيه إلى مصفوفة وكل كلمة على حدة | مصفوفة | `['أمير', 'الجوف', 'يشدّد', 'على', 'تطبيق', 'الإجراءات', ... 'الحمد']` |
| WordsCounts | رقم لعدد الكلمات في النص بعد التنظيف | رقم | `201` |

\* لعرض محتوى هذة الأعمدة، قم بإضافتها بعد الرابط `alriyadh.com/`

## تحليل استكشافي للبيانات

للمقالات التي تم نشرها بعد يوليو 2019.

### عدد المقالات يومياً

{% include NewsOverTimeFig.html %}

### عدد المقالات حسب الأيام

{% include DaysNewsfig.html %}

### عدد المقالات حسب النوع

المقالات نشرت في أكثر من نوع وهي كالتالي:

|**النوع**|**عدد المقالات**|
|:-----:|:-----:|
|الأخبار المصورة|29|
|مقالات اليوم|1112|
|متابعات|845|
|المنوعات|295|
|المحليات|5178|
|دنيا الرياضة|2347|
|الدولية|5339|
|الاقتصاد|2658|
|الأولـــى|192|
|الأخــيــرة|674|
|الرأي|522|
|كلمة الرياض|80|
|طــب|8|
|أخبار المناطق|859|
|خزامى الصحارى|24|
|سينما|3|
|صورة اليوم|10|
|فن|101|
|قول على قول|7|
|محطات متحركة|3|
|فيديو الرياض|1|
|ثقافة اليوم|235|
|تقارير دولية|4|
|تقارير رسومية|14|
|الأخبار الهامة|3|
|المجتمع الدولي|1|
|أدب الجمعة|1|
|الكاريكاتير|1|
|تحقيقات وتقارير|2|
|ثقافة السبت|2|
|اخر الثقافة|19|
|آخر الأخبار|1|
|نجوم الأمس الرياضي|1|

{% include NewsByCategoryfig.html %}

{% include NewsByCategoryHistofig .html %}

{% include NewsByCategoryPiefig.html %}

## تحميل البيانات

يمكن تحميل البيانات من صفحة المشروع في [قيت هب][github] أو [الرابط المباشر][direct]

## الترخيص والاستشهاد

### الترخيص
يمكن تحميل البيانات بشكل مجاني تحت ترخيص [Creative Commons Attribution 3.0 International license][CCA]

### الاستشهاد

عند استخدام البيانات يرجى الاستشهاد بالمصدر كالتالي:  
`Najwa Alghamdi and Ali Alohali, Saudi journalism in the age of COVID (2021).Submitted to Data in Brief.`


شكر خاص لجريدة الرياض على سماحهم لنا بنشر هذة البيانات


[najwa]: https://www.linkedin.com/in/dr-najwa-alghamdi-a77aa93/
[riyadh]: https://www.alriyadh.com/
[github]: https://github.com/alioh/AlRiyadh-Newspaper-Covid-Dataset/
[direct]: https://github.com/alioh/AlRiyadh-Newspaper-Covid-Dataset/raw/master/Alriyadh_News_Dataset.zip
[CCA]: https://creativecommons.org/licenses/by-nc/3.0/