###############################
#                             #
# Linear Regression           #
# Author: Stepheny Perez      #
#                             #
# This class uses the scikit  #
# python library to rn a      #
# linear regression.	      #
#                             #
###############################

#
# Note: Used Canvas notes as reference: 10_03_lr_implementation.pdf
#

from sklearn.linear_model import LinearRegression

class lr:
	def fit(self, X, y):
		self.model = LinearRegression()
		self.model.fit(X, y)

	def analysis(self):
		print ('Date slope: %.3f' % self.model.coef_[0])
		print ('Date intercept: %.3f' % self.model.intercept_)

	def predict(self, x):
		return self.model.predict(x)