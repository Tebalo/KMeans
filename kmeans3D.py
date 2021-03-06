# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 09:05:36 2021

@author: Bopaki
"""

import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 

#matplotlib inline
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

## Load Data
dfa = pd.read_csv("Mall_Customers.csv")
dfa = dfa[['Age','Annual Income (k$)','Spending Score (1-100)']]
print('Total Row : ', len(dfa))
## Feature Scaling
sc_dfa = StandardScaler()
dfa_std = sc_dfa.fit_transform(dfa.astype(float))
## Clustering with KMeans
kmeans = KMeans(n_clusters=3, random_state=42).fit(dfa_std)
labels = kmeans.labels_
new_dfa = pd.DataFrame(data = dfa_std, columns = ['Age','Annual Income (k$)','Spending Score (1–100)'])
new_dfa['label_kmeans'] = labels
fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(new_dfa.Age[new_dfa.label_kmeans == 0], new_dfa["Annual Income (k$)"][new_dfa.label_kmeans == 0], new_dfa["Spending Score (1–100)"][new_dfa.label_kmeans == 0], c='blue', s=100, edgecolor="green",linestyle="--")
ax.scatter(new_dfa.Age[new_dfa.label_kmeans == 1], new_dfa["Annual Income (k$)"][new_dfa.label_kmeans == 1], new_dfa["Spending Score (1–100)"][new_dfa.label_kmeans == 1], c='red', s=100, edgecolor="green",linestyle="--")
ax.scatter(new_dfa.Age[new_dfa.label_kmeans == 2], new_dfa["Annual Income (k$)"][new_dfa.label_kmeans == 2], new_dfa["Spending Score (1–100)"][new_dfa.label_kmeans == 2], c='green', s=100, edgecolor="green",linestyle="--")
ax.scatter(new_dfa.Age[new_dfa.label_kmeans == 3], new_dfa["Annual Income (k$)"][new_dfa.label_kmeans == 3], new_dfa["Spending Score (1–100)"][new_dfa.label_kmeans == 3], c='orange', s=100, edgecolor="green",linestyle="--")
ax.scatter(new_dfa.Age[new_dfa.label_kmeans == 4], new_dfa["Annual Income (k$)"][new_dfa.label_kmeans == 4], new_dfa["Spending Score (1–100)"][new_dfa.label_kmeans == 4], c='purple', s=100, edgecolor="green",linestyle="--")
centers = kmeans.cluster_centers_
ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='black', s=500);
plt.xlabel("Age")
plt.ylabel("Annual Income (k$)")
ax.set_zlabel('Spending Score (1–100)')
plt.show()