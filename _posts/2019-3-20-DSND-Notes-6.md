---  
layout: post
title: ملخص كورس علم البيانات - 6
---  

المخلص يغطي طريقة تقييم النموذج، مدى كفائته وكيف نحسنه.  
  
  


##### الفصل الأول - التعلم الموجَّه Supervised Learning  
##### الدرس الثامن - مقاييس تقييم النموذج Model Evaluation Metrics  

- **Train_test_split**: استخدمت طريقة في [الملخص الثاني](https://alioh.github.io/DSND-Notes-2) لتقسيم البيانات لقسمين، قسم للتدريب وقسم للإختبار، وتم إستعمال train_test_split لذلك، هذه أحدى طرق تقييم المودل. ولديها بعض الخصائص (متغيرات):  
    - **test_size**: حجم البيانات المراد إختبارها، اذا اعطيت 0.5، فانها تأخذ 50% من البيانات.  
    - **train_size**: نفس تعريف test_size ولكن لبيانات التدريب، ولا يحتاج لتعريفة وإعطائة قيمة اذا تم تعريف وإعطاء قيمة لحجم البيانات المراد إختبارها.  
    - **random_state**: عند تخصيص هذا المتغير، لا تتغير لدينا البيانات المفصولة في كل مرة نشغل فيها المودل. عندما لا يحدد، في كل مرة نشغل فيها المودل، تتغير لنا البيانات المفصلة لدينا بين البيانات التي تدرب والتي تُختَبر.  

##### مثال بايثون
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np

data = np.asarray(pd.read_csv('data.csv', header=None))
X = data[:,0:2]
y = data[:,2]

#   Test size is 25%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(y_test)

acc = accuracy_score(y_train, y_pred)
```
##### قياسات التصنيف Classification Metrics  

- **Confusion Matrix**: جدول يبين أداء المودل، ويكون كالتالي:  
![](https://alioh.github.io/images/2019-3-20/Confusion-matrix-example.png)  
اللون الأخضر في الجدول يعني البيانات التي تم توقعها وكانت نتائجها صحيحه، اللون الأحمر يعني نتائج كان توقعها خاطئ.  
  
- **Accuracy**: سبق ان تحدثت عنها في [الملخص الثالث](https://alioh.github.io/DSND-Notes-2)وهو يحدد عدد المرات التي كانت إجابة المُصنف (Classifier) صحيحه. وطريقة حسابه هي تابعه لل **Confusion Matrix**، وطريقة حسابه كالتالي:  
`[True Positives + True Negaitve/(Total Data)]`  

- **Precision**: سبق ان تحدثت عنها في [الملخص الثالث](https://alioh.github.io/DSND-Notes-2)، وقلنا انه يحدد نسبة مدى دقة المُصنف في توقعاته الصحيحه.  
`[True Positives/(True Positives + False Positives)]`  

- **Recall(sensitivity)**: أيضاً سبق ان تحدثت عنها في [الملخص الثالث](https://alioh.github.io/DSND-Notes-2)، وتقوم بحساب البيانات التي توقعناها أنها صحيحه من بين جميع البيانات التي توقعنا انها صحيحه.  
`[True Positives/(True Positives + False Negatives)]`  

- **F1 Score**: وأيضاً سبق ان تحدثت عنها في [الملخص الثالث](https://alioh.github.io/DSND-Notes-2)، وهي النتيجة الكاملة لكفائة ودقة المودل.  
`F1 Scode = 2 * ( (*Precision* * *Recall*) / (*Precision* + *Recall*) )`  
ويفضل إستخدامة بدل **Percision** و **Recall** كونهما أحياناً في بعض البيانات يقدما نتائج خاطئة ولا تدل فعلاً عن جودة المودل. [^1] [^2]  

- **F-beta**: معلومات مفصلة عنها [هنا](http://www.marcelonet.com/snippets/machine-learning/evaluation-metrix/f-beta-score)، ولكن ببساطة إذا اردنا ان نكرز أكثر على **Precision**، نعطيها رقماً بين 0 و 1، وإذا أردناها ان تركز على **Recall** أكثر، فنعطيها رقماً أعلى من 1.  

- **ROC Curve**: ويقوم هذا المقياس على تقييم مدى قدرة المودل على التوقع. [^3]  


##### قياسات الإنحدار الخطي Regression Metrics  

سبق أن شرحنا هذه النقاط في [الملخص الأول](https://alioh.github.io/DSND-Notes-1) وهي:
- **Mean Absolute Error**
- **Mean Squared Error**



-----
[العودة إلى ملخص كورس علم البيانات - 5](https://alioh.github.io/DSND-Notes-5/)   -   [الإنتقال إلى ملخص كورس علم البيانات - 7](https://alioh.github.io/DSND-Notes-7)  
  
  
[^1]: <https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9>
[^2]: <https://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/>
[^3]: <https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5>