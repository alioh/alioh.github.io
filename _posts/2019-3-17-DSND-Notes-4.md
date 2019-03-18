---  
layout: post
title: ملخص كورس علم البيانات - 4
---  

هذا الجزء سيكون عن آلة المتجهات الداعمة Support Vector Machines، متى تستخدم، والمتغيرات فيها وطريقة تكوين النموذج الخاص فيها ببايثون.   
  
  



###### الفصل الأول - الدرس السادس  
### آلة المتجهات الداعمة Support Vector Machines [^1]  
تعتبر من أكثر الخوارزميات استخداماً في التصنيف، وتقوم الخوارزمية بالبحث عن افضل طريقة لتقسيم البيانات. بحيث تحاول تكوين أكبر مسافه بين القيم.  
![](https://alioh.github.io/images/2019-2-11/1.jpg)  

#### Error Function  
Small margin > large error  
Large margin > small error  
وللتقليل من الاخطاء نستخدم Gradient Descent  

### Hyperparameters [^2]  

#### - The C parameter  
![](https://alioh.github.io/images/2019-3-17/c.png)  
* Large C: Focus on classifying points.  
* Small C: Focus on a large margin.  

#### - Kernels [^3] [^4]  
![](https://alioh.github.io/images/2019-3-17/kernels.png)  
* Polynomial Kernel  
* RBF Kernel  

#### - Gamma  
![](https://alioh.github.io/images/2019-3-17/gamma.png)  
للبيانات غير الخطية
* Large values of gamma tend to overfit.  
* Small values of gamma tend to underfit.  

#### - Degree  
![](https://alioh.github.io/images/2019-3-17/degree.png)  
تستخدم فقط في Polynomial Kernel.  



## مثال بايثون  
![](https://alioh.github.io/images/2019-3-17/data.png)  
قبل تشغيل المودل
```python
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

data = np.asarray(pd.read_csv('data.csv', header=None))
X = data[:,0:2]
y = data[:,2]

model = SVC(kernel='rbf', gamma=10, C=7)

model.fit(X, y)

y_pred = model.predict(X)
acc = accuracy_score(y, y_pred)
```
بعد التشغيل
![](https://alioh.github.io/images/2019-3-17/data1.png)  
  
  
-----
[العودة إلى ملخص كورس علم البيانات - 3](https://alioh.github.io/DSND-Notes-3/)   -   [الإنتقال إلى ملخص كورس علم البيانات - 5](https://alioh.github.io/DSND-Notes-5)  
  
  
[^1]: <https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/>
[^2]: <https://medium.com/all-things-ai/in-depth-parameter-tuning-for-svc-758215394769>
[^3]: <https://www.kdnuggets.com/2016/06/select-support-vector-machine-kernels.html>
[^4]: <https://towardsdatascience.com/support-vector-machines-svm-c9ef22815589>
