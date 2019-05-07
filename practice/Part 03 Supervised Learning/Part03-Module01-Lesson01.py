# -*- coding: utf-8 -*-
__author__ = "kaspar.s"
__date__ = '2019/4/28 21:17'

# TODO: Add import statements
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pysnooper

@pysnooper.snoop()
def Linear_Regression_in_scikit_learn_15():
	# Assign the dataframe to this variable.
	# TODO: Load the data
	bmi_life_data = pd.read_csv('bmi_and_life_expectancy.csv')

	# Make and fit the linear regression model
	#TODO: Fit the model and Assign it to bmi_life_model
	bmi_life_model = LinearRegression()
	bmi_life_model.fit(bmi_life_data[['BMI']],bmi_life_data[['Life expectancy']])
	# Make a prediction using the model
	# TODO: Predict life expectancy for a BMI value of 21.07931
	laos_life_exp = bmi_life_model.predict(np.array([21.07931]).reshape(-1,1))
	print(laos_life_exp)

@pysnooper.snoop()
def Multiple_Linear_Regression_17():
	from sklearn.linear_model import LinearRegression
	from sklearn.datasets import load_boston

	# Load the data from the boston house-prices dataset
	boston_data = load_boston()
	x = boston_data['data']
	y = boston_data['target']

	# Make and fit the linear regression model
	# TODO: Fit the model and assign it to the model variable
	model = LinearRegression()
	model.fit(x,y)

	# Make a prediction using the model
	sample_house = [[2.29690000e-01, 0.00000000e+00, 1.05900000e+01, 0.00000000e+00, 4.89000000e-01,
	                 6.32600000e+00, 5.25000000e+01, 4.35490000e+00, 4.00000000e+00, 2.77000000e+02,
	                 1.86000000e+01, 3.94870000e+02, 1.09700000e+01]]
	# TODO: Predict housing price for the sample_house
	prediction = model.predict(sample_house)

# Linear_Regression_in_scikit_learn_15()
Multiple_Linear_Regression_17()