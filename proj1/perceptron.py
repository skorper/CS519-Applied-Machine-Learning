# From notes:
# class Perceptron(object):
# 	# instance variables: eta (input by user to control how fast toconverge)
# 	# w (0...m)
# 	# num of iterations (from user)
# 	# error (array) implicit

# 	def _init(self, _eta=0.1, _iter=10):
# 		self.eta = _eta
# 		self.iter = _iter

# 	def learn(self, x, y):
# 		#generate m random w's
# 		generator = np,random.RandomState(_, _, _)
# 		self.w = generator.normal(_, _, m + 1)
# 		for i in range (self.iter):
# 			error[i] = 0
# 			for xi yi in zip(x, y):
# 				y_hat = predict(xi) #w transpose x (using numpy dot(w, xi)) will eventually replace with prediction function
# 				error[i] = error[i] + np.where(y_i == y_hat, 0, 1) #after each prediction, count how many are wrong
# 				for j in range (m):
# 					self.w[j] = self.w[j] + self.eta * (yi - yhat) * xi[j]

# 				self.w[0] = ...
# 			if error[i] == 0:  #probably need a threshold
# 				#then terminate

# 	def predict(self, x):
# 		input = np.dot(w, x)
# 		v = np.where(input > 0, 1, -1) #where function
# 		return v #predicted value

# Implementation: 

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
		#define 'm' (length of data attrs)
		self.m = x[1].size
		#generate m random w's
		generator = np.random.RandomState(1)
		self.w = generator.normal(0, 0.01, self.m + 1)
		self.error = []

		for i in range (self.iter):
			self.error.append(0)
			for xi, yi in zip(x, y):
				y_hat = self.predict(xi) #w transpose x (using numpy dot(w, xi))
				self.w[1:] += self.eta * (yi - y_hat) * xi
				
				# for j in range (self.m):
				# 	self.w[j] += ( self.eta * (yi - y_hat) )* xi

				self.error[-1] += np.where(yi == y_hat, 0, 1) #after each prediction, count how many are wrong
				self.w[0] = self.eta * (yi - y_hat)
			if self.error[-1] == 0.0:  #probably need a threshold then terminate. For now use 0
				break
			print (self.error[-1])

	def predict(self, x):
		input = np.dot(self.w[1:], x) + self.w[0]
		v = np.where(input > 0, 1, -1) #where function
		return v #predicted value