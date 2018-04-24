# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 15:16:17 2018

@author: ypare
"""

# =============================================================================
# Step 1: Import all libraries for data cleaning
# =============================================================================
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
#% matplotlib inline  #display plots instead of saving it in memory 

# =============================================================================
#  Step 2: Import File
# =============================================================================
df=pd.read_csv('titanic.csv')

# =============================================================================
# Handling Missing data
# =============================================================================
df=df.loc[df.Embarked.notnull(), :] #removes all records with null value
X=df.iloc[:,[2,4,5,6,7,9,11]].values
y=df.iloc[:,1].values
from sklearn.preprocessing import Imputer
imp=Imputer(missing_values='NaN', strategy='median') 
imp.fit(X[:, [2]])
X[:, [2]]=imp.transform(X[:, [2]])

# =============================================================================
# Handling Categorical data
# =============================================================================
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_1=LabelEncoder()
label_encoder_1.fit(X[:,1])
X[:,1]=label_encoder_1.transform(X[:,1])

label_encoder_6=LabelEncoder()
label_encoder_6.fit(X[:,6])
X[:,6]=label_encoder_6.transform(X[:,6])

onehot_encoder=OneHotEncoder(categorical_features=[1,6], sparse=False)
onehot_encoder.fit(X)
X=onehot_encoder.transform(X)
X=np.delete(X,onehot_encoder.feature_indices_[:-1],1)

# =============================================================================
# Split data into training and test
# =============================================================================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)

# =============================================================================
# Scaling Data
# =============================================================================
from sklearn.preprocessing import StandardScaler
#formula= value-mean/std dev

sc=StandardScaler()
sc.fit(X_train)
X_train=sc.transform(X_train)
X_test=sc.transform(X_test) #transformations are always made using model trained on training data

# =============================================================================
# Building Logistic Regression Model
# =============================================================================
from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression()
log_reg.fit(X_train,y_train)
# =============================================================================
# Prediction using Test Data
# =============================================================================

y_pred=log_reg.predict(X_test) #predicts class
y_pred_prob=log_reg.predict_proba(X_test) [:,1] # [:,1]-calculates probability of class 1(survived)
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

