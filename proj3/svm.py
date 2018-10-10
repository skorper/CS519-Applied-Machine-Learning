################################
#                              #
# svm.py Simple Vector Machine #
# Author: Stepheny Perez       #
#                              #
# This class uses the scikit   #
# python library to create an  #
# SVM classifier.              #
#                              #
################################

# Note: In order to create this class I used the following references:
#	- 03_02_svm.pdf (from Canvas)
#	- http://scikit-learn.org/stable/modules/svm.html

from sklearn.svm import SVC

# svm = SVC(kernel=rbf, random_state=1, gamma=0.2, C=1.0)
# svm.fit(X_train_std, y_train)

class SVM(object):

	def __init__(self, _gamma=0.2, _C=1.0):
		self.gamma = _gamma
		self.C = _C

	def learn(self, x, y):
		self.svm = SVC(kernel='rbf', random_state=1, gamma=self.gamma, C=self.C)
		self.svm.fit(x, y)

	def predict(self, x, y):
		y_pred = self.svm.predict(x)
		error = (y != y_pred).sum()
		print ("Misclassifications: " + str(error) + "/" + str(len(y_pred)) + " = " + str(error/len(y_pred) * 100) + "%")
