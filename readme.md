# Important NLP concepts
### resources
[Coursera NLP course] (https://www.coursera.org/learn/classification-vector-spaces-in-nlp)

### Logistics Regression
In this, we will classify tweets in to positive or negative sentiments using Logistic regression. Let us first understand supervised machine learning:
#### Supervised machine learning   
In supervised machine learning, you usually have an input X, which goes in to your prediction function to get you ŷ.  
You can then compare your prediction with the true value YY. This gives you your cost which you use to update the parameters θ (theta).  
The following image, summarizes the process. 
![](images/supervise_machine_learning.PNG)
#### Vocabulary and feature extraction
Given a tweet, or some text, you can represent it as a vector of dimension VV, where VV corresponds to your vocabulary size.  
If you had the tweet "I am happy because I am learning NLP", then you would put a 1 in the corresponding index for any word  
in the tweet, and a 0 otherwise.  

As you can see, as VV gets larger, the vector becomes more sparse. Furthermore, we end up having many more features and  
end up training \thetaθ VV parameters. This could result in larger training time, and large prediction time.  
~[](images/vocab_and_feature_extraction.PNG)  

