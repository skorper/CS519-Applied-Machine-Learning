# ML 8.27

# 1. Random number generator from numpy
#
# numpy random uniform (min/max?)
# numpy random normal (gaussian?) (standard deviation, scale, size)

# 2. Zip

# for x in zip(xlist)
# for x_i, y_i in zip(x y)

# Dot (from numpy)

#just does dot product

# numpy.dot(x, y) // x T y 
# can use something else, but recommends using dot because it's faster

# 4. Numpy where()

# 5. Paandas package read_csv [data frame data structure] iloc and values

# Perceptron

#when the data is not linearly seperable the error may go up and down and it will never converge

import numpy as np

class Perceptron(object):
	# instance variables: eta (input by user to control how fast toconverge)
	# w (0...m)
	# num of iterations (from user)
	# error (array) implicit

	def __init__(self, _eta=0.1, _iter=10):
		self.eta = _eta
		self.iter = _iter

	def learn(self, x, y):
		#define 'm'
		self.m = x.shape[1]
		#generate m random w's
		generator = np.random.RandomState(1)
		self.w = generator.normal(0, 0.01, self.m + 1)
		self.error = []

		for i in range (self.iter):
			self.error.append(0)
			for xi, yi in zip(x, y):
				y_hat = self.predict(xi) #w transpose x (using numpy dot(w, xi)) will eventually replace with prediction function
				self.error[i] = self.error[i] + np.where(yi == y_hat, 0, 1) #after each prediction, count how many are wrong
				for j in range (self.m):
					self.w[j] = self.w[j] + self.eta * (yi - y_hat) * xi[j]

				self.w[0] = self.eta * (yi - y_hat) #TODO fix this value
				if self.error[i] == 0:  #probably need a threshold then terminate
					break
			print (self.error)
	def predict(self, x):
		input = np.dot(self.w[1:], x)
		v = np.where(input > 0, 1, -1) #where function
		return v #predicted value