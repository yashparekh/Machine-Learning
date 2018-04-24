# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 09:56:04 2018

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
df = pd.read_csv("car_dataset.csv")
X = df.iloc[:,:-1].values
y= df.iloc[:,-1].values
# =============================================================================
# Step 4: Handling Missing Values. Missing values are either replaced with mean or median 
# =============================================================================
from sklearn.preprocessing import Imputer
imp=Imputer(missing_values='NaN', strategy='median')
imp.fit(X[:, [1]]) 
X[:, [1]]=imp.transform(X[:, [1]])
# =============================================================================
# Handling Categorical data
# =============================================================================
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_2=LabelEncoder()
label_encoder_2.fit(X[:,2])
X[:,2]=label_encoder_2.transform(X[:,2])

label_encoder_3=LabelEncoder()
label_encoder_3.fit(X[:,3])
X[:,3]=label_encoder_3.transform(X[:,3])

label_encoder_4=LabelEncoder()
label_encoder_4.fit(X[:,4])
X[:,4]=label_encoder_4.transform(X[:,4])

label_encoder_5=LabelEncoder()
label_encoder_5.fit(X[:,5])
X[:,5]=label_encoder_5.transform(X[:,5])

label_encoder_11=LabelEncoder()
label_encoder_11.fit(X[:,11])
X[:,11]=label_encoder_11.transform(X[:,11])

onehot_encoder=OneHotEncoder(categorical_features=[2,3,4,5,11], sparse=False)
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
# Building Lineaer Regression Model with variable selection
#For regression we normally use f_regression scoring based on p-value
#For classification we use chi2, f_classif for score_func
# =============================================================================
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression,SelectPercentile,SelectFromModel

#lr_model=Pipeline(steps=[
#        ('feature selection',SelectKBest(score_func=f_regression, k=5)),
#        ('linear regression',LinearRegression())
#        ])

#lr_model=Pipeline(steps=[
#        ('feature selection',SelectPercentile(score_func=f_regression, percentile=50)),
#        ('linear regression',LinearRegression())
#        ])
from sklearn.ensemble import RandomForestRegressor
random_forest_model=RandomForestRegressor(n_estimators=100)

lr_model=Pipeline(steps=[
        ('feature selection',SelectFromModel(estimator=random_forest_model)),
        ('linear regression',LinearRegression())
        ])

#Here model may not give good performance metrics as random forest does not used scaled data. To improve 
#performance, give unscaled data to random forest and use normalize=2 in Linear Regression()

lr_model.fit(X_train,y_train)
# =============================================================================
# Prediction using Test Data
# =============================================================================

y_pred=lr_model.predict(X_test)

# =============================================================================
# Regression Metrics
# =============================================================================

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mean_absolute_error(y_test,y_pred)
np.sqrt(mean_squared_error(y_test,y_pred)) #rmse/ used to penalize large values of error
r2_score(y_test,y_pred)
