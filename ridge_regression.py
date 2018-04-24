# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 13:56:20 2018

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
df = pd.read_csv("cruise_ship_info.csv")
X = df.iloc[:,1:-1].values
y= df.iloc[:,-1].values
# =============================================================================
# Step 4: Handling Missing Values. Missing values are either replaced with mean or median 
# =============================================================================
from sklearn.preprocessing import Imputer
imp=Imputer(strategy='median')  # ctrl+I inside paranthesis---created a model to replace missing values with median
#imputer fn does not work with categorical data
imp.fit(X[:, [1,2]]) #model is trained to replace missing values in first and second columns with median
#imp.statistics_ provides the calculated median for columns 1 and 2 for the training model0
X[:, [1,2]]=imp.transform(X[:, [1,2]])

# =============================================================================
# Step 5: Handling Categorical data
# =============================================================================

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#LabelEncoder assigns numerical value to each category. Eg 0 to male, 1 to female, 2 to others
#OneHotEncoder assigns binary values to each label. Eg [1,0,0] for 0, [0,1,0] for 1, [0,0,1] for 2

label_encoder=LabelEncoder()
label_encoder.fit(X[:,0])
X[:,0]=label_encoder.transform(X[:,0])

onehot_encoder=OneHotEncoder(categorical_features=[0], sparse=False)
onehot_encoder.fit(X)
X=onehot_encoder.transform(X)
X=np.delete(X,onehot_encoder.feature_indices_[:-1],1) #will delete first column of each categorical variable/ here 1 indicates delete column
#dummy variable trap/to remove multicollinearity due to dummy variable from the model. use either this or fit_intercept=0 in LinearRegression()
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
# Building Linear Regression Model
# =============================================================================
from sklearn.linear_model import Ridge
rd=Ridge()
rd.fit(X_train,y_train)

# =============================================================================
# Prediction using Test Data
# =============================================================================

y_pred=rd.predict(X_test)

# =============================================================================
# Regression Metrics
# =============================================================================

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mean_absolute_error(y_test,y_pred)
np.sqrt(mean_squared_error(y_test,y_pred)) #rmse/ used to penalize large values of error
r2_score(y_test,y_pred) #model explains 95% of variance in input variables
#calculate adjusted r^2 from r^2 
