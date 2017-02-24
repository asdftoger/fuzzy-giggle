# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 11:48:18 2017

@author: John
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import seaborn as sns

from scipy.cluster.hierarchy import linkage,dendrogram
from sklearn.preprocessing import MaxAbsScaler


X = pd.read_csv('./2014_USelection.csv')
X= X.dropna(axis = 0)
X = X.replace(',','',regex = True)

#y_party = X.State[X.iloc[:,1]<X.iloc[:,2]]
y = X.State.values
X = X.drop(axis = 1,labels = ['State'])

#for col in X.columns:
#    X[col] = pd.to_numeric(X[col])


scle = MaxAbsScaler()
scle.fit(X)
X = scle.transform(X)

plt.figure(figsize=(15,10))
merging = linkage(X,method = 'complete')

dendrogram(merging,labels = y
           ,leaf_rotation=90,leaf_font_size = 10)

plt.xlabel('State Code')
plt.margins(0.05)
plt.show()

