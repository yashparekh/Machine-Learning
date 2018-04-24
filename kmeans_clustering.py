# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 10:41:04 2018

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
df = pd.read_csv("Customers.csv")
X = df.iloc[:,-2:].values
#no y variable as we are not predicting or classifying anything

# =============================================================================
# Scaling Data
# =============================================================================
from sklearn.preprocessing import StandardScaler
#formula= value-mean/std dev

sc=StandardScaler()
sc.fit(X)
X=sc.transform(X)

# =============================================================================
# Finding best value of K for K-means clustering model
# =============================================================================
from sklearn.cluster import KMeans

wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,n_init=10,max_iter=300)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_) #kmeans.inertia_ provides within the group error

plt.figure(figsize=(10,8))
plt.plot(range(1,11),wcss)
plt.title('Elbow Plot')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# =============================================================================
# Building K-means Model
# =============================================================================

kmeans=KMeans(n_clusters=5, n_init=10,max_iter=300)
y_pred=kmeans.fit_predict(X) #fitting and predicting data together

# =============================================================================
# Plotting our cluster - Case of two variables/predictors
# =============================================================================

x_axis = df.iloc[ : , -2]
y_axis = df.iloc[ : , -1]

centroids = sc.inverse_transform(kmeans.cluster_centers_) #unscaling data to original form

#plt.figure(figsize = (10, 8))
plt.scatter(x_axis[y_pred == 0], y_axis[y_pred==0], color = "red", label = 'C1')
plt.scatter(x_axis[y_pred == 1], y_axis[y_pred==1], color = "blue", label = 'C2')
plt.scatter(x_axis[y_pred == 2], y_axis[y_pred==2], color = "green", label = 'C3')
plt.scatter(x_axis[y_pred == 3], y_axis[y_pred==3], color = "magenta", label = 'C4')
plt.scatter(x_axis[y_pred == 4], y_axis[y_pred==4], color = "orange", label = 'C5')
plt.scatter(centroids[ : , 0], centroids[ : , 1], 
            color = "yellow", label = 'Centroids', s = 300)
plt.title('Customer Cluster')
plt.xlabel('Annual Income')
plt.ylabel('Spending')
plt.show()

