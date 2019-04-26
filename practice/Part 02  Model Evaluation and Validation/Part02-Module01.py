# -*- coding: utf-8 -*-
__author__ = "kaspar.s"
__date__ = '2019/4/23 21:08'

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm  import SVC
from sklearn.model_selection import train_test_split  # 分离函数
from sklearn.metrics import accuracy_score  # 检验准确度
import pysnooper

@pysnooper.snoop( )
def test01():
	data = pd.read_csv('2_class_data.csv')

	np_X = np.array(data[['x1','x2']])
	np_y = np.array(data['y'])

	classifier = LogisticRegression()
	classifier.fit(np_X,np_y)

	classifier1 = DecisionTreeClassifier()
	classifier1.fit(np_X, np_y)

	classifier2 = SVC(kernel='poly',degree=2,gamma=200)
	classifier2.fit(np_X, np_y)

@pysnooper.snoop()
def test02():
	data = np.asarray(pd.read_csv('data.csv', header=None))
	# Assign the features to the variable X, and the labels to the variable y.
	X = data[:, 0:2]
	y = data[:, 2]

	# Use train test split to split your data
	# Use a test size of 25% and a random state of 42
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)

	# Instantiate your decision tree model
	model = DecisionTreeClassifier()
	# TODO: Fit the model to the training data.
	model.fit(X_train,y_train)

	# TODO: Make predictions on the test data
	y_pred = model.predict(X_test)
	# TODO: Calculate the accuracy and assign it to the variable acc on the test data.
	acc = sum(y_pred == y_test)/len(y_test)
	acc1 = accuracy_score(y_test, y_pred)

# test01()
test02()