---  
layout: post
title: ملخص كورس علم البيانات - 2
---  

في الجزء الثاني من الملخص سأحاول أن تكون المعلومات أدق ولكن بشكل أقل، واحاول زيادة عدد الصور كونها أفضل طريقة للشرح (بالنسبة لي). بما أن الدروس (دسمة) بالمعلومات، من المتوقع أن يكون كل منشور مخصص لدرسين أو ثلاث. سأحاول أيضاً اضافة بعض المصطلحات التي اراها مهمه وشرحها. وبما أن الشرح باللغة الانجليزية، من الصعب إيجاد تعبير مناسب او ترجمة عربية مناسبة وصحيحة 100% لذا سأحاول إضافة التعريف أو الشرح الإنجليزي كمرجع في حال لم يكن الشرح بالعربية واضح.  

###### الفصل الأول - الدرس الثالث  
### خوارزمية بيرسيبترون Perceptron Algorithm  
تعتبر من أساسيات الشبكات العصبية Neural Networks. وتستخدم الخوارزمية للإجابة على الأسئلة بنعم أو لا، مثلاً هل هذه البريد مزعج أو لا. [^1] [^2]
![](https://alioh.github.io/images/2019-3-15/perceptron.png) 

------

###### الفصل الأول - الدرس الرابع  
### شجرة القرار Decision Tree
خوارزمية شجرة القرار ممكن ان تعمل على البيانات الرقمية والفئوية.  
تم شرحها بتفصيل اكثر [هنا](https://alioh.github.io/Machine-Learning-for-Everyone-3/).  

#### متغيرات الفصل في شجرة القرار
- عمق الفصل أو مداه Maximum Depth:  
(max_depth: The maximum number of levels in the tree)  
![](https://alioh.github.io/images/2019-2-11/11.png)  
في المثال السابق Maximum Depth = 2  
- الحد الادنى من البيانات للفصل Minimum number of samples to split:  
(min_samples_leaf: The minimum number of samples allowed in a leaf)  
![](https://alioh.github.io/images/2019-3-15/min-samples-split.png)  
- الحد الادنى للبيانات لكل ورقة Minimum number of samples per leaf:  
(min_samples_split: The minimum number of samples required to split an internal node)  
![](https://alioh.github.io/images/2019-3-15/screen-shot-2018-01-06-at-9.41.01-pm.png)

------

###### مصطلحات

One-hot encode: [^3]  
تحويل القيم الفئوية في البيانات إلى عواميد، في حال كانت القيمة من تلك الفئة، يوضع 1 عند ذلك العامود وصفر عند بقية العواميد، مثال:  
![](https://alioh.github.io/images/2019-3-15/one-hot-encoding.jpeg)

train_test_split: [^4]  
تستخدم لتقسيم البيانات إلى بيانات للتدريب وأخرى للإختبار، بعض المتغيرات المهمه فيها:
- test_size: حجم البيانات المراد إختبارها، اذا اعطيت 0.5، فانها تأخذ 50% من البيانات.
- train_size: نفس تعريف test_size ولكن لبيانات التدريب، ولا يحتاج لتعريفة وإعطائة قيمة اذا تم تعريف وإعطاء قيمة لحجم البيانات المراد إختبارها.
- random_state: "randomstate is basically used for reproducing your problem the same every time it is run. If you do not use a randomstate in train_test_split, every time you make the split you might get a different set of train and test data points and will not help you in debugging in case you get an issue." [^5]


[^1]: <https://towardsdatascience.com/what-the-hell-is-perceptron-626217814f53>
[^2]: <http://www.saedsayad.com/artificial_neural_network_bkp.htm>
[^3]: <https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f>
[^4]: <https://medium.com/@contactsunny/how-to-split-your-dataset-to-train-and-test-datasets-using-scikit-learn-e7cf6eb5e0d>
[^5]: <https://www.kaggle.com/questions-and-answers/49890>