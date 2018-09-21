# From notes:
# 
# #Adaline cost function: J(w) = 1/2 * (y - f(z))^2 //sum squared error
# class Adaline(object):
# 	def _init_(...):
# 		...
# 	def learn(self, x, y):
#
# 		# s1: initialize w using random gen
#
# 		for i in range(iter):
# 			#differences here
# 			#calculate net input
# 			net_input = np.dot(x, self.w[1:]) + self.w[0]
# 			error = y_net_input
# 			self.w[0] = self.eta.error.sum()
# 			self.w[1:] = elf.eta * x.t.dot(error)

# Implementation: (similar to Perceptron)

import numpy as np
class Adaline(object):

	def __init__(self, _eta=0.1, _iter=10):
		self.eta = _eta
		self.iter = _iter

	def learn(self, x, y):

		# s1: initialize w using random gen
		# this is the same as the perceptron class
		#define 'm' (length of data attrs)
		self.m = x[1].size
		#generate m random w's
		generator = np.random.RandomState(1)
		self.w = generator.normal(0, 0.01, self.m + 1)

		for i in range(self.iter):
			#differences here
			#calculate net input
			net_input = np.dot(x, self.w[1:]) + self.w[0]
			error = y - net_input
			self.w[0] += self.eta * error.sum()
			self.w[1:] += self.eta * x.T.dot(error)
			# Adaline cost function (sum squared error)
			cost = (error**2).sum() / 2.0
			print (cost)
		return self

	# same as perceptron prediction
	def predict(self, x):
		input = np.dot(self.w[1:], x) + self.w[0]
		v = np.where(input > 0, 1, -1) #where function
		return v #predicted value