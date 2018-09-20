# From notes;
#
# gen = np.RangomGenerator
# r = gen.permutation(n)
# x[r] = new vector which is shuffled
# #y[r] needs to match x[r]
# for i in range (iter):
# 	r = shuffle
# 	x = x[r]
# 	y = y[r]
# 	for xi , target in zip(x, y):
# 			# update weights
# 			zi = np.dot(xi, w[1]) + self.w[0]
# 			error = target - zi
# 			self.w[1:] = self.eta * xi.dot(error)
# 			self.w[0] = self.eta * error
# 	cost += 0.5 * error ** 2 

# Implementation: (similar to Adaline)

import numpy as np
import math
class Sgd(object):

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
		self.cost_ = []
		for i in range(self.iter):
			# shuffle training instances for more
			# stochastic effect
			r = generator.permutation(len(y))
			x = x[r]
			y = y[r]
			cost = []
			for xi, target in zip(x, y):
				zi = np.dot(xi, self.w[1:]) + self.w[0]
				error = target - zi
				self.w[1:] += self.eta * xi.dot(error)
				self.w[0] += self.eta * error
				#Getting an overflow error here with error squared
				cost.append(0.5 * np.power(error, 2))
			# average the costs from the training instances 
			# in this iteration (from notes)
			average_cost = sum(cost) / len(y)
			self.cost_.append(average_cost)
			print (sum(cost))
		return self

	# same as perceptron prediction
	def predict(self, x):
		input = np.dot(self.w[1:], x) + self.w[0]
		v = np.where(input > 0, 1, -1) #where function
		return v #predicted value