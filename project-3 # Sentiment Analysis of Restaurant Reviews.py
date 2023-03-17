#!/usr/bin/env python
# coding: utf-8

# # Project-3
# # Sentiment Analysis of Restaurant Reviews

# ## problem statement

# Normally, a lot of businesses are remained as failures due to lack of profit, lack of proper improvement measures. Mostly, restaurant owners face a lot of difficulties to improve their productivity. This project really helps those who want to increase their productivity, which in turn increases their business profits. This is the main objective of this project.
# 
# What the project does is that the restaurant owner gets to know about drawbacks of his restaurant such as most disliked food items of his restaurant by customer’s text review which is processed with ML classification algorithm(Naive Bayes)

# The purpose of this analysis is to build a prediction model to predict whether a review on the restaurant is positive or negative. To do so, we will work on Restaurant Review dataset, we will load it into predicitve algorithms Multinomial Naive Bayes, Bernoulli Naive Bayes and Logistic Regression. In the end, we hope to find a "best" model for predicting the review's sentiment.
Dataset: Restaurant_Reviews.tsv is a dataset from Kaggle datasets which consists of above 1000 reviews on a restaurant.

To build a model to predict if review is positive or negative, following steps are performed.

Importing Dataset
Preprocessing Dataset
Vectorization
Training and Classification
Analysis Conclusion Tools & Technologies Used:
NLTK
Machine Learning
Python
Tkinter/Flask
Mysql
Pandas
# In[1]:


# Importing the libraries


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


# read dataset


# In[4]:


# Loading the dataset
df = pd.read_csv('r_data.tsv', delimiter='\t')


# In[5]:


df


# In[6]:


df.shape


# In[7]:


df['Liked'].value_counts()


# In[8]:


# 1...Good review/positive
# 0 ..bad review /negative


# In[9]:


# data set is balanced


# In[10]:


s1=df['Liked'].value_counts()


# In[11]:


s1.plot(kind='bar')


# In[12]:


# missing values


# In[13]:


df.isnull().sum()


# In[14]:


# Text preprocessing


# Preprocessing Dataset Each review undergoes through a preprocessing step, where all the vague information is removed.
# 
# Removing the Stopwords, numeric and speacial charecters. Normalizing each review using the approach of stemming.

# In[15]:


import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re


# In[16]:


nltk.download('stopwords')


# In[17]:


# clean reviews
# remove special characters /numeric values
# remove stopwords
# stemming..to find stem word


# In[19]:


corpus=[]
for i in range(len(df)):
    # remove special characters and digits
    mystr=re.sub('[^A-Za-z\s]','',df['Review'][i])
    # lower case
    mystr=mystr.lower()
    # tokenization
    list1=mystr.split()
    # remove stopwords
    list2=[ i for i in list1 if i not in set(stopwords.words('english'))]
    # stemming
    ps=PorterStemmer()
    list3=[ ps.stem(i) for i in list2]
    # original string
    final=' '.join(list3)
    corpus.append(final)   


# In[20]:


corpus


# # Vectorization
# 
# From the cleaned dataset, potential features are extracted and are converted to numerical format. The vectorization techniques are used to convert textual data to numerical format. Using vectorization, a matrix is created where each column represents a feature and each row represents an individual review.

# In[21]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()


# In[24]:


X=cv.fit_transform(corpus).toarray()
y=df['Liked']


# In[25]:


X.shape


# In[26]:


# import pickle
# f=open('cv.pkl','wb')
# pickle.dump(cv,f)
# f.close()


# In[27]:


# cross validation 


# In[28]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                        test_size = 0.2, random_state = 42)


# In[30]:


#model training


# Training and Classification Further the data is splitted into training and testing set using Cross Validation technique. This data is used as input to classification algorithm.
# 
# Classification Algorithms:
# 
# Algorithms like Decision tree, Support Vector Machine, Logistic Regression, Naive Bayes were implemented and on comparing the evaluation metrics two of the algorithms gave better predictions than others.
# 
# Multinomial Naive Bayes Bernoulli Naive Bayes Logistic Regression

# In[31]:


# Multinomial Naive Bayes


# In[33]:


from sklearn.naive_bayes import MultinomialNB
clf1= MultinomialNB()
clf1.fit(X_train, y_train)
# Predicting the Test set results
y_pred = clf1.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print ("Confusion Matrix:\n",cm)

# Accuracy, Precision and Recall
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
print('Classification Report\n', classification_report(y_test, y_pred))
score1 = accuracy_score(y_test,y_pred)
score2 = precision_score(y_test,y_pred)
score3= recall_score(y_test,y_pred)
print("\n")
print("Accuracy is ",round(score1*100,2),"%")
print("Precision is ",round(score2,2))
print("Recall is ",round(score3,2))


# In[37]:


import seaborn as sns
sns.heatmap(cm,annot=True,fmt='d')
plt.xlabel('Truth')
plt.ylabel('Predicted')
plt.show()


# In[38]:


# #Bernoulli NB


# In[39]:


from sklearn.naive_bayes import BernoulliNB
clf2= BernoulliNB()
clf2.fit(X_train, y_train)
# Predicting the Test set results
y_pred = clf2.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print ("Confusion Matrix:\n",cm)

# Accuracy, Precision and Recall
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
print('Classification Report\n', classification_report(y_test, y_pred))
score1 = accuracy_score(y_test,y_pred)
score2 = precision_score(y_test,y_pred)
score3= recall_score(y_test,y_pred)
print("\n")
print("Accuracy is ",round(score1*100,2),"%")
print("Precision is ",round(score2,2))
print("Recall is ",round(score3,2))


# In[ ]:




