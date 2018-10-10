##############################################################
#                                                            #
# CS 519 Project 3                                           #
# Author: Stepheny Perez                                     #
#                                                            #
# This is the main script which the user directly calls to   #
# test the classifier classes I've written for this project. #
#                                                            #
##############################################################

import sys
import numpy as np 
import pandas as pd

from sklearn.datasets import load_digits
from timeit import default_timer as timer
from perceptron import SingleLayerPerceptron #had to rename to avoid conflict w/scikit lib
from svm import SVM
from lr import LR
from dt import DT
from knn import KNN

# Parse/validate arguments

if len(sys.argv) != 3:
    print ("Incorrect usage. python main.py <algorithm> <dataset>")
    exit()

# Load data (only two options in this case!)
x = None
y = None
if sys.argv[2] == 'digits':
	x, y = load_digits(return_X_y=True)
elif sys.argv[2] == 'absent':
	# Source: https://archive.ics.uci.edu/ml/datasets/Absenteeism+at+work
	df = pd.read_csv("absent.csv")
	x = df.iloc[0:, 0:len(df.columns) - 1].values
	y = df.iloc[0:, len(df.columns) - 1].values
	y = list(map(int, y))
	# True class = less than 5 hours absent

	# np.where is being weird so loop through and do this manually...
	for i in range (len(y)):
		if y[i] < 5:
			y[i] = 1
		else:
			y[i] = -1
else:
	print ("Incorrect usage. Dataset options are \'digits\' and \'absent\'")
	exit()

percent_training 	= 70
training_size 		= int(len(x)*(percent_training / 100))
model 				= None

if sys.argv[1] == 'perceptron':
    model = SingleLayerPerceptron()
elif sys.argv[1] == 'svm':
    model = SVM()
elif sys.argv[1] == 'lr':
    model = LR()
elif sys.argv[1] == 'dt':
    model = DT()
elif sys.argv[1] == 'knn':
    model = KNN()
else:
    print ("Incorrect algorithm. Options are perceptron, svm, lr, dt, or knn ")
    exit()

# Normalize data

x_std = np.copy(x)
x_std[:, 0] = (x[:, 0] - x[:, 0].mean()) / x[:, 0].std()
x_std[:, 1] = (x[:, 1] - x[:, 1].mean()) / x[:, 1].std()

# Train model

start = timer()
model.learn(x[0:training_size], y[0:training_size])
end = timer()
print ("Training time: " + str(end-start))

# Predict from testing data and print error percentage
start = timer()
model.predict(x[training_size:], y[training_size:])
end = timer()
print ("Prediction time: " + str(end-start))