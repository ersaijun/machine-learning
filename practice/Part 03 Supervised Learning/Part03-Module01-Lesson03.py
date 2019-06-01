# -*- coding: utf-8 -*-
__author__ = "kaspar.s"
__date__ = '2019/5/18 16:24'

# Decision Trees in sklearn

# Import statements
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import pysnooper


# Read the data.
data = np.asarray(pd.read_csv('Part03-Module01-Lesson03.csv', header=None))
# Assign the features to the variable X, and the labels to the variable y.
X = data[:,0:2]
y = data[:,2]

# TODO: Create the decision tree model and assign it to the variable model.
# You won't need to, but if you'd like, play with hyperparameters such
# as max_depth and min_samples_leaf and see what they do to the decision
# boundary.
model = DecisionTreeClassifier(max_depth=5)

# TODO: Fit the model.
model.fit(X,y)
# TODO: Make predictions. Store them in the variable y_pred.
y_pred = model.predict(X)

# TODO: Calculate the accuracy and assign it to the variable acc.
acc = sum(y_pred == y)/y.size *100
acc1 = accuracy_score(y,y_pred)
print(acc1)