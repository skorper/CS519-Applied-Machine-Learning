###############################
#                             #
# RANSAC Regression           #
# Author: Stepheny Perez      #
#                             #
# This class uses the scikit  #
# python library to rn a      #
# ransac regression.	      #
#                             #
###############################

#
# Note: Used Canvas notes as reference: 10_03_lr_implementation.pdf
#

from sklearn.linear_model import RANSACRegressor
import numpy as np

class rr:
	def __init__(self, _max_trials=100, _min_samples=50, _loss='absolute loss', _residual_threshold=5.0):
		self.max_trials = _max_trials
		self.min_samples = _min_samples
		self.loss = _loss
		self.residual_threshold = _residual_threshold

	def fit(self, X, y):
		self.model = RANSACRegressor(random_state=1)
		self.model.fit(X, y)
		line_y = self.model.predict(X[:,np.newaxis])

	def analysis(self):
		print ('Date slope: %.3f' % self.model.coef_[0])
		print ('Date intercept: %.3f' % self.model.intercept_)

	def predict(self, x):
		return self.model.predict(x)