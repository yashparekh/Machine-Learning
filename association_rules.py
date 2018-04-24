# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 13:46:13 2018

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
df = pd.read_csv(filepath_or_buffer="Market_Basket_Optimisation.csv", header=None)
transactions=[]

for i in range(0,7501):
    transactions.append([str(df.values[i,j]) for j in range(0,20)])
    
# =============================================================================
# Training Apriori on Dataset
# =============================================================================

from apyori import apriori
rules=apriori(transactions, min_support=0.003, min_confidence= 0.2, min_lift=3, min_length=2) 
#min_length is the minimum number of items in the list

# =============================================================================
# Visualizing the rules
# =============================================================================
results=list(rules)
rules_list=[]
for i in range(0,len(results)):
    rules_list.append('Rule:\t'+str(results[i][0])+'\nSupport:\t'+str(results[i][1]))