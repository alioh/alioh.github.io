---  
layout: post
title: ملخص كورس علم البيانات - 8
---  

هذا الملخص عن طريقة تطبيق Gradient Descent في الشبكات العصبية وطريقة تدريبها وتجنب فرط التخصيص.  
  
  


##### الفصل الثاني - التعلم العميق Deep Learning  
##### الدرس الثاني - Implementing Gradient Descent  
فكرة Gradient Descent هي طريقة تحسين النتائج في كل خطوة والتكرار حتى تصل لنقطة تكون فيها النتائج مرضية. [^1]  

##### Multilayer Perceptrons  
كيفية عملية الحساب داخل الخلية العصبية كالتالي: [^2]  
![](https://alioh.github.io/images/2019-3-29/NeuralNet.png)  
مثال عملي: [الجزء الأول](https://stevenmiller888.github.io/mind-how-to-build-a-neural-network/)، [الجزء الثاني](http://stevenmiller888.github.io/mind-how-to-build-a-neural-network-part-2/)  

##### Backpropagation  
طريقة أخرى لتعليم الخلية العصبية، مثال عملي مع شرح بأكواد بايثون [هنا](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/)  


##### الفصل الثاني - التعلم العميق Deep Learning  
##### الدرس الثالث - Training Neural Networks  

##### طرق لتجنب فرط التخصيص Overfitting  

* Early Stopping [^3]  
* Regularization [^4]  
* Dropout [^5]  

##### متغيرات  
* Learning Rate: كلما كانت اقل، كلما زادت فرصة الوصول أقل نقطة Error في المودل.  
* Momentum: [^6] [^7]  


-----
[العودة إلى ملخص كورس علم البيانات - 7](https://alioh.github.io/DSND-Notes-7/)   -   [الإنتقال إلى ملخص كورس علم البيانات - 9](https://alioh.github.io/DSND-Notes-9)  
  
  
[^1]: [TowardsDataScience](https://towardsdatascience.com/gradient-descent-in-a-nutshell-eaf8c18212f0)
[^2]: [VisualStudioMagazine](https://visualstudiomagazine.com/articles/2013/05/01/neural-network-feed-forward.aspx)
[^3]: [TowardsDataScience](https://towardsdatascience.com/early-stopping-2f92c29ce0ae)
[^4]: [AnalyticsVidhya](https://www.analyticsvidhya.com/blog/2018/04/fundamentals-deep-learning-regularization-techniques/)
[^5]: [Amarbudhiraja](https://medium.com/@amarbudhiraja/https-medium-com-amarbudhiraja-learning-less-to-learn-better-dropout-in-deep-machine-learning-74334da4bfc5)
[^6]: [TowardsDataScience](https://towardsdatascience.com/stochastic-gradient-descent-with-momentum-a84097641a5d)
[^7]: [VisualStudioMagazine](https://visualstudiomagazine.com/articles/2017/08/01/neural-network-momentum.aspx)