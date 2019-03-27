---  
layout: post
title: ملخص كورس علم البيانات - 7
published: false
---  

هذا الملخص بداية للفصل الثاني بعنوان التعلم العميق ويعطي لمحة بسيطة عن الشبكات العصبية.  
  
  


##### الفصل الثاني - التعلم العميق Deep Learning  
##### الدرس الأول - مدخل إلى الشبكات العصبية Introduction to Neural Networks  
عمل الشبكات العصبية مشابه لحد كبير لعمل عقل الإنسان، وفيها الآلة تتعلم من الأنماط التي تلاحظها في البيانات التي نعطيها. تستخدم خوارزميات مختلفه في ذلك وتعمل عملية حسابية تستخرج منها افضل النتائج لإتخاذ القرارات. التعلم العميق جزء من تعلم الآلة لذا يجب ان نعرف وتكون لدينا معلومات اساسية عن تعلم الآلة وأساسياتها. شرحت اغلبها في الملخصات السابقة.  
![](https://alioh.github.io/images/2019-2-10/1.png)  

##### Discrete vs Continuous  
يقصد بـDiscrete هي البيانات التي ليست متسلسلة مثلاً `[2,6,9,15]` ونقوم بعملية حسابيه لتحويلها لبيانات متسلسلة من 0 إلى 1، البيانات يفضل انت تكون دائماً متسلسلة في الشبكات العصبية.  ``sigmoid(x) = 1/(1+e^-x)``
![](https://alioh.github.io/images/2019-2-27/DataConversionExamples.GIF)  

##### The Softmax Function  
عندما تكون لدينا أكثر من نتيجتين، مثلاً في الأمثلة السابقة كنا نتوقع رسائل البريد مزعجه أو لا، ماذا لو أردنا ان يخبرنا المودل إذا كان البريد مهم أو لا أو إذا كان مزعجاً.  
الآن لدينا ثلاث نتائج:
* البريد مهم
* البريد مزعج
* البريد غير ذلك
نستخدم القانون التالية لتحويل النتائج Scores إلى إحتمالات Probabilites: [^1]  
![](https://alioh.github.io/images/2019-2-27/maxresdefault.jpg)  

##### مثال بايثون  
في المثال التالي طريقة حساب Softmax  
```python
import numpy as np

def softmax(L):
    expL = np.exp(L)
    sumExpL = sum(expL)
    result = []
    for i in expL:
        result.append(i*1.0/sumExpL)
    return result
```

##### One-Hot Encoding  
تحدثت في [الملخص الثاني](https://alioh.github.io/DSND-Notes-2/) عن هذه العملية.  
![](https://alioh.github.io/images/2019-3-15/one-hot-encoding.jpeg)  

##### Maximum Likelihood  
هي طريقة لزيادة الـProbability وعندما تكون الـ Probability أعلى، يدل أن نتائج المودل ممتازة. [^2]  

##### Cross-Entropy  
تقيم اداء المودل، كلما كان الرقم أقل كل ما كان أفضل. [مثال](https://www.youtube.com/watch?v=tRsSi_sqXjI)  

##### مثال بايثون  
```python
import numpy as np

def cross_entropy(Y, P):
    Y = np.float_(Y)
    P = np.float_(P)
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))
```

##### الإنحدار اللوجستي Logistic Regression  
سبق ان تحدثت عنه بشكل مفصل في [تعلم الآلة للجميع -3](https://alioh.github.io/Machine-Learning-for-Everyone-3/)



-----
[العودة إلى ملخص كورس علم البيانات - 6](https://alioh.github.io/DSND-Notes-6/)   -   [الإنتقال إلى ملخص كورس علم البيانات - 8](https://alioh.github.io/DSND-Notes-8)  
  
  
[^1]: <https://www.youtube.com/watch?v=lvNdl7yg4Pg>
[^2]: <https://towardsdatascience.com/probability-concepts-explained-maximum-likelihood-estimation-c7b4342fdbb1>
