# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 13:01:17 2018

@author: ypare
"""

# =============================================================================
# Import Libraries
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# =============================================================================
# Import Dataset
# =============================================================================
df = pd.read_csv("Restaurant_Reviews.tsv", delimiter="\t", quoting=2) #quoting removes double quotes

# =============================================================================
# Cleaning Data
# =============================================================================
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus=[]
for i in range(0,1000):
    review=re.sub('[^a-zA-Z]',' ',df['Review'][i]) #Only take alphabets, replace other characters with space
    review=review.lower()
    review=review.split() #created a list of words by splitting the sentence
    ps=PorterStemmer() #used for stemming. stemming helps in transforming different forms of words having the same meaning. eg changes loved to love
    review=[ps.stem(words) for words in review if not words in stopwords.words('english')]
    # for each word in review, if the word is not in the stopwords list then keep that word otherwise remove it 
    review=' '.join(review) #join each word with a space'
    corpus.append(review)

# =============================================================================
# Building a Bag of Words Model
# =============================================================================
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X=cv.fit_transform(corpus).toarray()
y=df.iloc[:,-1].values
# =============================================================================
# Split data into training and test
# =============================================================================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)
# =============================================================================
# Building Naive Bayes Model
# =============================================================================
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(X_train,y_train)
# =============================================================================
# Prediction using Test Data
# =============================================================================

y_pred=nb.predict(X_test) #predicts class
y_pred_prob=nb.predict_proba(X_test) [:,1] # [:,1]-calculates probability of class 1(survived)
# =============================================================================
# Classification Metrics
# =============================================================================
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)

from sklearn.metrics import roc_curve, roc_auc_score
fpr,tpr,threshold=roc_curve(y_test,y_pred_prob)
auc_score=roc_auc_score(y_test,y_pred_prob)
plt.title('Receiving Operating Characteristics')
plt.plot(fpr,tpr,'b',label='AUC=%0.2f'%auc_score)
plt.legend(loc = 'lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

from sklearn.metrics import f1_score
f1_score(y_test, y_pred)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))