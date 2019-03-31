---  
layout: post
title: ملخص كورس علم البيانات - 9
---  

هذا الملخص عن مكتبة Keras في بايثون. وهي مكتبة متخصصه في الشبكات العصبية.  
  
  


##### الفصل الثاني - التعلم العميق Deep Learning  
##### الدرس الرابع - Keras  
أحدى أشهر مكاتب الشبكات العصبية في بايثون هي Keras، وليست وحدها الموجودة، يوجد أيضاً TensorFlow, Caffe, Scikit-learn وغيرها الكثير.  
الدرس الخاص بالمكتبة عملي، وسأضع بعض الأكواد لشرح كيفية تكوين وتصميم شبكة عصبية بمكتبة Keras. [^1]  

```python
import numpy as np
from keras.utils import np_utils
import tensorflow as tf
tf.python.control_flow_ops = tf

np.random.seed(42)

#   البيانات (المدخلات) لدينا بالشكل التالي
#   0 | 0
#   0 | 1
#   1 | 0
#   1 | 1
X = np.array([[0,0],[0,1],[1,0],[1,1]]).astype('float32')

#   ونتائج كل سطر من المدخلات مخزن في y
#   0
#   1
#   1
#   0
y = np.array([[0],[1],[1],[0]]).astype('float32')

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten

# One-hot encoding للنتائج
y = np_utils.to_categorical(y)

#   إنشاء المودل
xor = Sequential()
#   إضافة لاير ب32 نود يستقبل 2 كمدخلات
xor.add(Dense(32, input_dim=2))
#   إضافة لير آخر بدالة tanh
xor.add(Activation("tanh"))
#   إضافة لاير جديد بنودين
xor.add(Dense(2))
#   إضافة لير آخر بدالة sigmoid
xor.add(Activation("sigmoid"))
#   نحتاج لجمع وتشغيل المودل قبل إدخال البيانات له
#   loss هي طريقة حساب التناقض بين النتائج المتوقعة والنتائج الحقيقة
#   optimizer او المحسنات، توجد انواع مختلفة وتحتاج لتعريفها
#   metrics طريقة تقيم النتائج
xor.compile(loss="categorical_crossentropy", optimizer="adam", metrics = ['accuracy'])

#   لعرض شكل المودل النهائي
xor.summary()

#   نقوم بتشغيل المودل على البيانات
#   nb_epoch عدد مرات تشغيل المودل على البيانات
#   verbose طريقة عرض النتائج
history = xor.fit(X, y, nb_epoch=1000, verbose=0)

#   تقيم النتائج
score = xor.evaluate(X, y)
print("\nAccuracy: ", score[-1])

#   عرض التوقعات
print("\nPredictions:")
print(xor.predict_proba(X))
```
* Sequential [^2]  
* Dense [^3]
* Activation [^4]  
* nb_epoch [^5]
* verbose [^6]  

أمثلية عملية: [1](https://towardsdatascience.com/how-to-build-a-neural-network-with-keras-e8faa33d0ae4)، [2](https://towardsdatascience.com/building-a-deep-learning-model-using-keras-1548ca149d37)


-----
[العودة إلى ملخص كورس علم البيانات - 8](https://alioh.github.io/DSND-Notes-8/)   -   [الإنتقال إلى ملخص كورس علم البيانات - 10](https://alioh.github.io/DSND-Notes-10)  
  
  
[^1]: [KDnuggets](https://www.kdnuggets.com/2018/06/keras-4-step-workflow.html)
[^2]: [Coursera](https://www.coursera.org/lecture/ai/sequential-models-in-keras-RBbLP)
[^3]: [Keras](https://keras.io/layers/core/#dense)
[^4]: [Keras](https://keras.io/activations/)
[^5]: [Stackoverflow](https://stackoverflow.com/a/47905435/2022948)
[^6]: [Stackoverflow](https://stackoverflow.com/a/44907684/2022948)