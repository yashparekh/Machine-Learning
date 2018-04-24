# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 11:16:40 2018

@author: ypare
"""

# =============================================================================
# Importing Libraries
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# Importing our dataset
# =============================================================================

df = pd.read_csv(filepath_or_buffer = "Customers.csv")
X = df.iloc[ : , -2:].values

# =============================================================================
# Using Dendograms to find optimal number of cluster
# =============================================================================

import scipy.cluster.hierarchy as sch

dendrograms = sch.dendrogram(sch.linkage(X, method = 'ward' ))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()
# =============================================================================
# Building Hierarchical Clustering model
# =============================================================================

from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters= 5)
y_pred = hc.fit_predict(X)


# =============================================================================
# Plotting our cluster - Case of two variables/predictors
# =============================================================================

x_axis = df.iloc[ : , -2]
y_axis = df.iloc[ : , -1]


#plt.figure(figsize = (10, 8))
plt.scatter(x_axis[y_pred == 0], y_axis[y_pred==0], color = "red", label = 'C1')
plt.scatter(x_axis[y_pred == 1], y_axis[y_pred==1], color = "blue", label = 'C2')
plt.scatter(x_axis[y_pred == 2], y_axis[y_pred==2], color = "green", label = 'C3')
plt.scatter(x_axis[y_pred == 3], y_axis[y_pred==3], color = "magenta", label = 'C4')
plt.scatter(x_axis[y_pred == 4], y_axis[y_pred==4], color = "orange", label = 'C5')
plt.title('Customer Cluster')
plt.xlabel('Annual Income')
plt.ylabel('Spending')
plt.show()
