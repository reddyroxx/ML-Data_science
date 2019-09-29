#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 12:34:21 2019

@author: rakesh
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from subprocess import check_output

train = pd.read_csv('final_t1.csv')
test = pd.read_csv('final_test.csv')

corr = train.corr()
plt.figure(figsize=(8,8))
sns.heatmap(corr)
plt.yticks(rotation=0, size=7)
plt.xticks(rotation=90, size=7)
plt.show()


rel_vars = corr.price[(corr.price > 0.2)]
rel_cols = list(rel_vars.index.values)

corr2 = train[rel_cols].corr()
plt.figure(figsize=(8,8))
hm = sns.heatmap(corr2, annot=True, annot_kws={'size':10})
plt.yticks(rotation=0, size=10)
plt.xticks(rotation=90, size=10)
plt.show()

X = train[rel_cols[:-1]].iloc[:,0:].values
y = train.iloc[:, -1].values


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state = 0)