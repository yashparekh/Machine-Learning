# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 12:29:09 2018

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
df = pd.read_csv("Wine.csv")
X = df.iloc[:,0:13].values
y= df.iloc[:,-1].values
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
X_test=sc.transform(X_test)

# =============================================================================
# Experimenting with PCA
# =============================================================================
#from sklearn.decomposition import PCA
#pca1=PCA()
#X_train=pca1.fit(X_train)
#explained_variance=np.cumsum(pca1.explained_variance_ratio_)
#
#plt.figure(figsize=(6,4))
#plt.plot(explained_variance)
#plt.show()

#pca can be applied to dataset only once in one python session

# =============================================================================
# Applying PCA
# =============================================================================

from sklearn.decomposition import PCA
pca=PCA(n_components=6)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)

# =============================================================================
# Building Logistic Regression Model
# =============================================================================
from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression()
log_reg.fit(X_train,y_train)
# =============================================================================
# Prediction using Test Data
# =============================================================================

y_pred=log_reg.predict(X_test)
# =============================================================================
# Classification Metrics
# =============================================================================
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)