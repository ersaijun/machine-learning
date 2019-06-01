# -*- coding: utf-8 -*-
__author__ = "kaspar.s"
__date__ = '2019/6/1 22:04'

# Support Vector Machines in sklearn

import pysnooper

# Import statements
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Read the data.
data = np.asarray(pd.read_csv('Part03-Module01-Lesson05.csv', header=None))
# Assign the features to the variable X, and the labels to the variable y.
X = data[:,0:2]
y = data[:,2]

# TODO: Create the model and assign it to the variable model.
# Find the right parameters for this model to achieve 100% accuracy on the dataset.
model = SVC(kernel='rbf',gamma=30)
# C: The C parameter.
# kernel: The kernel. The most common ones are 'linear', 'poly', and 'rbf'.
# degree: If the kernel is polynomial, this is the maximum degree of the monomials in the kernel.
# gamma : If the kernel is rbf, this is the gamma parameter.
# TODO: Fit the model.
model.fit(X,y)
# TODO: Make predictions. Store them in the variable y_pred.
y_pred = model.predict(X)

# TODO: Calculate the accuracy and assign it to the variable acc.
acc = accuracy_score(y_pred,y)
print(acc)