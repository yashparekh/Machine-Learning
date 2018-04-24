# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
# =============================================================================
# Step 1: Import all libraries for data cleaning
# =============================================================================
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
% matplotlib inline  #display plots instead of saving it in memory 

# =============================================================================
#  Step 2: Import File
# =============================================================================
df=pd.read_csv('data_cleaning.csv')
# =============================================================================
# Step 3: Separate data into input(X) and output(y) variables
# =============================================================================
X=df.iloc[:, [1,2,3]].values #df.iloc[:, 1:-1].values --- columns 1,2,3 except the last column
y=df.iloc[:,-1].values

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

#at a time only one categorical data can be transformed using label encoder.In case of one hot encoder, just change the column number

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