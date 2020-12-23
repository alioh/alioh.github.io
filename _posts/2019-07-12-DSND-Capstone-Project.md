---  
layout: post
title: Data Science Nanodegree Capstone Project
icon: ☕
---  

This project (Capstone Project) is part of Udacity's Data Scientist Nanodegree program    
  


  
  
  
  
<h2 align="left">Starbucks Best Offers Predictor / Analysis</h2>
  
![](https://alioh.github.io/images/2019-7-12/Starbucks_Recyclable_Cups.jpg)
  
<p align="center" size="1" color='gray'>Copyright: monticello/123RF.<font size="1" color="white"> e</font> </p>   

<h3 align="left">Project Overview</h3>
<p align="left">
In this project, I will try to find how Starbucks customers use the app, and how well is the current offers system. I will also see who should the app target in promotions. The data sets used in this project contains simulated data that mimics customer behavior on the Starbucks rewards mobile app. From it, we can understand the costumers' behavior and it might help us make better decisions.<font size="1" color="white"> e</font>  </p> 
<h3 align="left">Problem Statement</h3>
<p align="left">The problem we have here is that we don't want to give any customer our offers. We want to give only those who we think will be able to complete the offer. Giving an offer to someone we know he/she probably will not be able to complete it is a waste of time and resources that can be given to someone who we know will complete it. I will approach this problem by first cleaning up the data, then doing some exploratory analysis and see who are my most valuable customers after that I will create a model to help us predicting feature customers and which type of offer should we give them.<font size="1" color="white"> e</font>  
<p align="left">My goal for this project is predicting which kind of offers, Buy One Get One Free (BOGO), Discount or informational is better to give a current customer by only knowing his/her age, gender, income and the amount they are paying.<font size="1" color="white"> e</font>  
</p> 
<h3 align="left">Metrics</h3>
<p align="left">
The metric I used this project is accuracy. Since we have a simple classification problem, I will use accuracy to evaluate my models. We want to see how well our model by seeing the number of correct predictions vs total number of predictions. For the different models I used in this project, I checked the accuracy my training and testing data sets and decided which to choose based on it.<font size="1" color="white"> e</font> </p>

<h3 align="left">Analysis</h3>  

<img src="https://alioh.github.io/images/2019-7-12/pestle-analysis-of-starbucks.jpg">
  
<p align="center" size="4" color='gray'>Copyright: <a href="https://www.flickr.com/photos/opengridscheduler/16604095887/">opengridscheduler</a><font size="1" color="white"> e</font> </p>   

<h2 align="left">Business understanding</h2>  
<p align="left">
My objective here is to find patterns and show when and where to give specific offer to a specific customer. Main users of this kind of applications are Starbucks employees and analysts. The plan in this project to have questions and answer them with data visualization. Tha data is provided by Starbucks contains simulated data that mimics customer behavior.<font size="1" color="white"> e</font>  
</p> 

<h2 align="left">Data Exploration / Understanding</h2>  
<p align="left">
In this project we were given 3 files. Before I start analyzing we have to explore and see what is the data we have. We need to check if it is clean or not, if each column have the right type that the data tell, for example if the data in column called price is saved as string, we need to convert it to number to help us in the analysis if we want to find the sum for example, having it as string will not return the total of that column. Similar thing goes to dates saved as strings. <font size="1" color="white"> e</font>  
</p> 
<p align="left">The data we have is provided by Starbucks. Here is a quick breakthrough of how the data looks like:<font size="1" color="white"> e</font></p>  
 <ul dir='ltr' align="left">
  <li align="left">portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)  </li>
  <li align="left">profile.json - demographic data for each customer  </li>
  <li align="left">transcript.json - records for transactions, offers received, offers viewed, and offers completed  </li>
</ul>
<p align="left">Here is the schema and explanation of each variable in the files:<font size="1" color="white"> e</font></p>
<p align="left"><b>portfolio.json</b> - <i>10 rows, 6 columns</i>.<font size="1" color="white"> e</font></p>
 <ul dir='ltr' align="left">
    <li align="left">id (string) - offer id</li>
    <li align="left">offer_type (string) - type of offer ie BOGO, discount, informational</li>
    <li align="left">difficulty (int) - minimum required spend to complete an offer</li>
    <li align="left">reward (int) - reward given for completing an offer</li>
    <li align="left">duration (int) - time for offer to be open, in days</li>
    <li align="left">channels (list of strings)</li>
</ul>

<p align="left"><b>profile.json</b> - <i>17000 rows, 5 columns</i>.<font size="1" color="white"> e</font></p>
 <ul dir='ltr' align="left">
    <li align="left">age (int) - age of the customer</li>
    <li align="left">became_member_on (int) - date when customer created an app account</li>
    <li align="left">gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)</li>
    <li align="left">id (str) - customer id</li>
    <li align="left">income (float) - customer's income</li>
</ul>

<p align="left"><b>transcript.json</b> - <i>306534 rows, 4 columns</i>.<font size="1" color="white"> e</font></p>
 <ul dir='ltr' align="left">
    <li align="left">event (str) - record description (ie transaction, offer received, offer viewed, etc.)</li>
    <li align="left">person (str) - customer id</li>
    <li align="left">time (int) - time in hours since start of test. The data begins at time t=0</li>
    <li align="left">value - (dict of strings) - either an offer id or transaction amount depending on the record</li>
</ul>
<h2 align="left">Data preparation / Wrangling</h2>  
<p align="left">In this part I did a lot of changes to the three tables. Here are the changes I made:<font size="1" color="white"> e</font></p>
<h4 align="left">portfolio.json</h4>
<p align="left"><b align="left">channels column</b>: this hold a list of the channels where the offer is delivered. and to fix it I one-hot-encoded it to look like this.<font size="1" color="white"> e</font></p>  

<img src="https://alioh.github.io/images/2019-7-12/one-hot-encode.jpg">

<h4 align="left">profile.json</h4>
<p align="left">
The gender and income column have NaN values. For gender, NaN were converted to NA. For income, NaN were replaced by the mean.<font size="1" color="white"> e</font></p>  

<h4 align="left">transcript.json</h4>
<p align="left">
Similar to what we saw before in portfolio channels column, here the value column holds dictionary of offer id, amount, offer_id and reward. To fix this I will do the same think I did before with channels, one-hot-encoding, and I will combine offer_id and offer id since both means the same thing. The final result looks like this.<font size="1" color="white"> e</font></p>  

<img src="https://alioh.github.io/images/2019-7-12/one-hot-encode2.jpg">

<h2 align="left">Analysis</h2>

<h3 align="left">A. Univariate Exploration</h3>
<h4 align="left">What are the most common values for each column in each data frame</h4>
<img src="https://alioh.github.io/images/2019-7-12/a1-1.jpg">

<p align="left">
For age,  we can see that most of ages in our profile data frame falls in-between 40 and 80. We already notice one outlier which is 118. Our median is around 58 years old.<font size="1" color="white"> e</font></p>  

<img src="https://alioh.github.io/images/2019-7-12/a2-1.jpg"> 
<img src="https://alioh.github.io/images/2019-7-12/a2-2.jpg"> <img src="https://alioh.github.io/images/2019-7-12/a2-3.jpg">

<p align="left">The first bar chart tell us that we have a lot profile in the adult age group, ages between 21 and 64.<font size="1" color="white"> e</font></p>  


<h4 align="left">What is the average income for Starbucks customers</h4>

<img src="https://alioh.github.io/images/2019-7-12/a1-2.jpg">

<p align="left">For the income, most of their income are between 50k and 78k. The exact number for average income is 65404.<font size="1" color="white"> e</font></p>  

<h4 align="left">What is the average age for Starbucks customers</h4>
<img src="https://alioh.github.io/images/2019-7-12/a1-1.jpg">

<p align="left">From what we saw in the first question, our average age is around 58.<font size="1" color="white"> e</font></p>  

<h4 align="left">What is the most and least common promotion</h4>
<img src="https://alioh.github.io/images/2019-7-12/a5-1.jpg">

<p align="left">The offer ID 'fafdcd668e3743c1bb461111dcafc2a4' is the most common with number of completion equal to 5317. The least common offer is '4d5c57ea9a6940dd891ad53e9dbe8da0' with total of 3331 completion.<font size="1" color="white"> e</font></p>  

<img src="https://alioh.github.io/images/2019-7-12/a5-2.jpg">

<p align="left">The most common types of offers is BOGO and Discounts.<font size="1" color="white"> e</font></p>  

<h4 align="left">Who are the most loyal customer - most transcripts</h4>
<p align="left">Here are a list of the most loyal customers (customers who spends a lot of money on offers/transactions).<font size="1" color="white"> e</font></p>  

<img src="https://alioh.github.io/images/2019-7-12/a7-1.jpg">


<h4 align="left">What are the most events we have in our transcripts</h4>
<img src="https://alioh.github.io/images/2019-7-12/a8-1.jpg">

<p align="left">Transaction have the most amount of rows in the transcript data frame with around 140k, almost half of our data frames total.<font size="1" color="white"> e</font></p>  

<h3 align="left">B. Multivariate Exploration</h3>
<h4 align="left">What is the most common promotion for children, teens, young adult, adult and elderly customers</h4>
<img src="https://alioh.github.io/images/2019-7-12/b1-1.jpg">

<p align="left">
NA = Transactions. We can see that most of our customers falls in the adult and elderly group age. And they prefer Buy One Get One and Discount offers than informational offers.<font size="1" color="white"> e</font></p>  

<h4 align="left">From profiles, which get more income, males or females</h4>
<img src="https://alioh.github.io/images/2019-7-12/b2-1.jpg">

<p align="left">The graph above shows that income median (the white dot) for females (around 70k) is higher than males (around 60k) we can also see that for females the income spreads from 40k to 100k. For males most of them around 40k to 70k which close to median.<font size="1" color="white"> e</font></p>  

<h4 align="left">What is the gender distribution in the transcript data frame</h4>
<img src="https://alioh.github.io/images/2019-7-12/b3-1.jpg">

<p align="left">Total number of males records 155690, and total number of female records is 113101.<font size="1" color="white"> e</font></p>  
<p align="left">From the two graphs above we can see that males received offers more than females. Both genders seems to reflect on those offers similarly. Around half of offers received were viewed by both genders, but it seems that females would complete those offers more than males. The numbers are:<font size="1" color="white"> e</font></p>  

<img src="https://alioh.github.io/images/2019-7-12/b3-2.jpg">
<img src="https://alioh.github.io/images/2019-7-12/b3-3.jpg">

<p align="left">For Females:<font size="1" color="white"> e</font></p>  
<p align="left">Total transcripts is: 113101.<font size="1" color="white"> e</font></p>  
<p align="left">Number of bogo offers: 27619, 43.34% of total.<font size="1" color="white"> e</font></p>  
<p align="left">Number of discount offers: 26652, 41.83% of total.<font size="1" color="white"> e</font></p>  
<p align="left">Number of informational offers: 9448, 14.83% of total.<font size="1" color="white"> e</font></p>  
<p align="left">Number of offer completed: 15477, 56.37% of total offers received.<font size="1" color="white"> e</font></p>  
<p align="left">Number of offer received: 27456, 43.09% of total.<font size="1" color="white"> e</font></p>  
<p align="left">Number of offer viewed: 20786, 32.62% of total.<font size="1" color="white"> e</font></p>  
<p align="left">Number of transaction: 49382, 43.66% of total.<font size="1" color="white"> e</font></p>  

<p align="left">For Males:<font size="1" color="white"> e</font></p>  
<p align="left">Total transcripts is: 155690.<font size="1" color="white"> e</font></p>  
<p align="left">Number of bogo offers: 35301, 42.58% of total.<font size="1" color="white"> e</font></p>  
<p align="left">Number of discount offers: 34739, 41.91% of total.<font size="1" color="white"> e</font></p>  
<p align="left">Number of informational offers: 12856, 15.51% of total.<font size="1" color="white"> e</font></p>  
<p align="left">Number of offer completed: 16466, 43.18% of total offers received.<font size="1" color="white"> e</font></p>  
<p align="left">Number of offer received: 38129, 46.0% of total.<font size="1" color="white"> e</font></p>  
<p align="left">Number of offer viewed: 28301, 34.14% of total.<font size="1" color="white"> e</font></p>  
<p align="left">Number of transaction: 72794, 64.36% of total.<font size="1" color="white"> e</font></p>  

<p align="left">The numbers above shows that males receive offers more than females by 9% and their transaction is 19% more too, which tells that they both more than females. Regarding offers, Males and Females received the same amount of <b>BOGO</b> and <b>discount</b> offers.<font size="1" color="white"> e</font></p>  

<h4 align="left">Who takes long time to achieve each promotion goal and from which gender, age, income</h4>

<p align="left">The mean time it takes a customer to complete an offer is around <b>16 days</b> (390 hours).<font size="1" color="white"> e</font></p>  

<h4 align="left">Which type of promotions each gender likes - offer_type</h4>
<img src="https://alioh.github.io/images/2019-7-12/b6-1.jpg">

<p align="left">We can see that both genders like bogo and discount offers and they have the same reaction to informational offers, they both seem to be not interested to it.<font size="1" color="white"> e</font></p>  

<h4 align="left">From each offer received by customer, how many they completed</h4>
<img src="https://alioh.github.io/images/2019-7-12/b7-1.jpg">

<p align="left">For Females:<font size="1" color="white"> e</font></p>  
<p align="left">Number of offer completed: 15477, 56.37% of total offers received.<font size="1" color="white"> e</font></p>  
<p align="left">Number of offer received: 27456, 43.09% of total.<font size="1" color="white"> e</font></p>  
<p align="left">Number of offer viewed: 20786, 32.62% of total.<font size="1" color="white"> e</font></p>  
<p align="left">Number of transaction: 49382, 43.66% of total.<font size="1" color="white"> e</font></p>  

<p align="left">For Males:<font size="1" color="white"> e</font></p>  
<p align="left">Number of offer completed: 16466, 43.18% of total offers received.<font size="1" color="white"> e</font></p>  
<p align="left">Number of offer received: 38129, 46.0% of total.<font size="1" color="white"> e</font></p>  
<p align="left">Number of offer viewed: 28301, 34.14% of total.<font size="1" color="white"> e</font></p>  
<p align="left">Number of transaction: 72794, 64.36% of total.<font size="1" color="white"> e</font></p>  
<p align="left">Females completed <b>56%</b> of the offers they received, it is <b>13%</b> more than males, but males made more transactions than females, <b>64%</b> to <b>43%</b>.<font size="1" color="white"> e</font></p>  


<h2 align="left">Model</h2>
<p align="left">In this part, I tried to make a model that can identify which kind of offers we should give a customer. Because my model will guess the offer_type, I will only get those transcripts with offer id's. So I will ignore all transactions without offer id for now. <font size="1" color="white"> e</font> </p>  
<p align="left">Our features here are:<font size="1" color="white"> e</font> </p>  
 <ul dir='ltr' align="left">
    <li align="left">Event. (Will be replaced from categorical to numerical)</li>
    <li align="left">Time. (normalized)</li>
    <li align="left">Offer_id. (Will be replaced from categorical to numerical)</li>
    <li align="left">Amount. (normalized)</li>
    <li align="left">Reward. (normalized)</li>
    <li align="left">Age_group. (Will be replaced from categorical to numerical)</li>
    <li align="left">Gender. (Will be replaced from categorical to numerical).</li>
    <li align="left">Income. (normalized)</li>
</ul>
<p align="left">And my target is offer_type. For my target, I will replace texts with numbers. Where BOGO = 1, discount = 2, informational = 3.<font size="1" color="white"> e</font> </p> 
<p align="left">Here is how the final data frame looks like before modeling:<font size="1" color="white"> e</font> </p> 

<img src="https://alioh.github.io/images/2019-7-12/df_model.jpg">

<p align="left">The shape of my features and labels was:<font size="1" color="white"> e</font> </p> 
 <ul dir='ltr' align="left">
    <li align="left">Training Features Shape: (125685, 8)</li>
    <li align="left">Training Labels Shape: (125685,)</li>
    <li align="left">Testing Features Shape: (41896, 8)</li>
    <li align="left">Testing Labels Shape: (41896,)</li>
</ul>
<p align="left">Now for the modeling part, I tried six different models and this is the results:<font size="1" color="white"> e</font> </p>  

<img src="https://alioh.github.io/images/2019-7-12/model_result.jpg"> 

<p align="left">From the previous table, we can see that we scored 100% accuracy in the training and testing data sets on 4 models. To avoid over fitting I will choose <b>Logistic Regression</b> since it got good results 65% on training and 80% on testing data sets. <b>Logistic Regression</b> is better used here since we have few binomial outcomes (BOGO = 1, discount = 2, informational = 3). It is good here because we have good amount of data to work with.<font size="1" color="white"> e</font> </p>

<h2 align="left">Conclusion</h2>  
<p align="left">In this project, I tried to analyze and make model to predict the best offer to give a Starbucks customer. First I explored the data and see what I have to change before start the analysis. Then I did some exploratory analysis on the data after cleaning. From that analysis I found out that most favorite type of offers are <b>Buy One Get One</b> (BOGO) offers and <b>Discount</b> offers. I digged deep to see who and what type of customers we have and noticed that <b>Females</b> tend to complete offers more than males with <b>56%</b> completion of the offers they received. Where <b>Males</b> completed only <b>43.18%</b> from the offers they received. But our current data shows that we gave <b>Males</b> more offers since they have more transactions than <b>Females</b> with total number of <b>72794</b> transactions, where females only had <b>49382</b> transactions.
In conclusion, the company should give more offers to <b>Females</b> than <b>Males</b> since they have more completed offers. And they should focus more on <b>BOGO</b> and <b>Discount</b> offers since they are the one that tend to make customers buy more.<font size="1" color="white"> e</font></p>  


<h2 align="left">Improvements</h2> 
<p align="left">I think I got to a point where we have good results and we understand the data we have very well. But to make our results even better, I would try to improve my data collection and fix issues I have with NaN values. I will also try to get even more data like location and when the transaction were completed, which branch and what time of the day. All these data can help us know when and where to give our offers. Also having more data is always good think to help us improve our model results.<font size="1" color="white"> e</font></p>  

<p align="left">
To see more detailed analysis with numbers and codes, check the project Github repository <a href='https://github.com/alioh/DSND-Capstone'>here</a>.<font size="1" color="white"> e</font> </p> 
