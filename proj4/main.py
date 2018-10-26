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
import numpy 	as np 
import pandas 	as pd

from sklearn.preprocessing 		import StandardScaler
from sklearn.model_selection 	import train_test_split

from sklearn.datasets 	import load_digits
from timeit 			import default_timer as timer
from sklearn.metrics 	import r2_score
from sklearn.metrics 	import mean_squared_error as mse

from lr 	import lr
from rr 	import rr
from ridge 	import ridge
from lasso 	import lasso

# Load data
# Soure: ML Book: pg 313

df = pd.read_csv('housing.data.txt', header=None, sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.head()

x = df[['RM']].values
y = df['MEDV'].values


# Data preprocessing

y2d = y[:, np.newaxis]

sc_x = StandardScaler()
sc_y = StandardScaler()

sc_x.fit(x)
sc_y.fit(y.reshape(-1,1))

x_std = sc_x.transform(x)
y_std = sc_y.transform(y2d).flatten()

x_train, x_test, y_train, y_test = train_test_split(x_std, y_std, test_size=0.3, random_state=0)

# Run Linear Regression

model = None

# lr()
# model.fit(x_std, y_std)
# model.analysis()

if sys.argv[1] == 'linear':
    model = lr()
elif sys.argv[1] == 'ransac':
    model = rr()
elif sys.argv[1] == 'ridge':
    model = ridge()
elif sys.argv[1] == 'lasso':
    model = lasso()
else:
    print ("Incorrect algorithm. Options are linear, ransac, ridge, and lasso")
    exit()

# Fit model

start = timer()
model.fit(x_train, y_train)
end = timer()
print ("Fit time: " + str(end - start))
model.analysis()

# Predict

y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

print('MSE train: %.3f, test: %.3f' % (mse(y_train, y_train_pred), mse(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))