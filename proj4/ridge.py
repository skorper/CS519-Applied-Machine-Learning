###############################
#                             #
# Ridge Regression            #
# Author: Stepheny Perez      #
#                             #
# This class uses the scikit  #
# python library to rn a      #
# rigge regression.	          #
#                             #
###############################

#
# Note: Used Canvas notes as reference: 10_03_lr_implementation.pdf
#

from sklearn.linear_model import Ridge

class ridge:
	def __init__(self, _alpha=1.0):
		self.alpha = _alpha
	def fit(self, X, y):
		self.model = Ridge(alpha=self.alpha)
		self.model.fit(X, y)

	def analysis(self):
		print ('Date slope: %.3f' % self.model.coef_[0])
		print ('Date intercept: %.3f' % self.model.intercept_)

	def predict(self, x):
		return self.model.predict(x)