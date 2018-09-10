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

class Perceptron(object):
	# instance variables: eta (input by user to control how fast toconverge)
	# w (0...m)
	# num of iterations (from user)
	# error (array) implicit

	def _init(self, _eta=0.1, _iter=10):
		self.eta = _eta
		self.iter = _iter

	def learn(self, x, y):
		#generate m random w's
		generator = np,random.RandomState(_, _, _)
		self.w = generator.normal(_, _, m + 1)
		for i in range (self.iter):
			error[i] = 0
			for xi yi in zip(x, y):
				y_hat = predict(xi) #w transpose x (using numpy dot(w, xi)) will eventually replace with prediction function
				error[i] = error[i] + np.where(y_i == y_hat, 0, 1) #after each prediction, count how many are wrong
				for j in range (m):
					self.w[j] = self.w[j] + self.eta * (yi - yhat) * xi[j]

				self.w[0] = ...
			if error[i] == 0:  #probably need a threshold
				#then terminate

	def predict(self, x):
		input = np.dot(w, x)
		v = np.where(input > 0, 1, -1) #where function
		return v #predicted value