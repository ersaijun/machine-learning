# -*- coding: utf-8 -*-
__author__ = "kaspar.s"
__date__ = '2019/4/23 22:21'

import pandas as pd
import numpy as np
from sklearn.linear_model import  LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
import pysnooper
import matplotlib.pyplot as plt


@pysnooper.snoop()
def test01():
	# 06. Detecting Overfitting and Underfitting with Learning Curves
	X2, y2 = get_data()
	# TODO: Uncomment one of the three classifiers, and hit "Test Run"
	# to see the learning curve. Use these to answer the quiz below.

	### Logistic Regression
	# estimator = LogisticRegression()

	### Decision Tree
	# estimator = GradientBoostingClassifier()

	### Support Vector Machine
	estimator = SVC(kernel='rbf', gamma=1000)

	draw_learning_curves(X2, y2,estimator,100)
# It is good to randomize the data before drawing Learning Curves
def randomize(X, Y):
	permutation = np.random.permutation(Y.shape[0])
	X2 = X[permutation,:]
	Y2 = Y[permutation]
	return X2, Y2

def get_data():
	data = pd.read_csv('p02m03.csv')
	X = np.array(data[['x1', 'x2']])
	y = np.array(data['y'])

	np.random.seed(50)
	X2, y2 = randomize(X, y)
	return X2,y2

def draw_learning_curves(X, y, estimator, num_trainings):
	train_sizes, train_scores, test_scores = learning_curve(
	    estimator, X, y, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, num_trainings))

	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)

	plt.grid()

	plt.title("Learning Curves")
	plt.xlabel("Training examples")
	plt.ylabel("Score")

	plt.plot(train_scores_mean, 'o-', color="g",
	         label="Training score")
	plt.plot(test_scores_mean, 'o-', color="y",
	         label="Cross-validation score")


	plt.legend(loc="best")

	plt.show()


# 09. Grid Search in sklearn
# 寻找最优参数
@pysnooper.snoop()
def test02():
	from sklearn.model_selection import GridSearchCV
	from sklearn.metrics import make_scorer
	from sklearn.metrics import  f1_score

	parameters = {'kernel':['poly','rbf'],'C':[0.1,1,10]}
	score = make_scorer(f1_score)

	estimator = SVC()
	grid_obj = GridSearchCV(estimator,parameters,scoring=score)
	X,y = get_data()
	grid_fit = grid_obj.fit(X,y)

	best_clf = grid_fit.best_estimator_
	pass
# test01()
test02()