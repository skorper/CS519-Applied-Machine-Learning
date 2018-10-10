###############################
#                             #
# Perceptron                  #
# Author: Stepheny Perez      #
#                             #
# This class uses the scikit  #
# python library to create a  #
# perceptron classifier.      #
#                             #
###############################

# Note: In order to create this class I used the following references:
#	- 03_01_scikit_learn_general_steps.pdf (from Canvas)
#	- http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html
#	- My own proj 1 submission (https://github.com/skorper/CS519-Applied-Machine-Learning/blob/master/proj1/perceptron.py)

from sklearn.linear_model import Perceptron

class SingleLayerPerceptron(object):

	def __init__(self, _eta=0.1, _iter=10):
		self.eta = _eta
		self.iter = _iter

	def learn(self, x, y):
		self.perceptron = Perceptron(max_iter=self.iter, eta0=self.eta, random_state=1)
		self.perceptron.fit(x, y)

	def predict(self, x, y):
		y_pred = self.perceptron.predict(x)
		error = (y != y_pred).sum()
		print ("Misclassifications: " + str(error) + "/" + str(len(y_pred)) + " = " + str(error/len(y_pred) * 100) + "%")
