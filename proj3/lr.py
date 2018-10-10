##################################
#                                #
# lr.py Logistic Regression      #
# Author: Stepheny Perez         #
#                                #
# This class uses the scikit     #
# python library to create a     #
# logistic regression classifier #
#                                #
##################################

# Note: In order to create this class I used the following references:
#	- 03_05_logistic_regression.pdf (from Canvas)
#	- http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

from sklearn.linear_model import LogisticRegression

# lr = LogisticRegression(C=100.0, random_state=1)
# lr.fit(X_train_std, y_train)
# y_pred_lr = lr.predict(X_test_std)
# y_pred_lr_prob = lr.predict_proba(X_test_std)
# print(y_pred_lr_prob.shape)
# print(y_pred_lr_prob)

class LR(object):

	def __init__(self, _C=100.0):
		self.C = _C

	def learn(self, x, y):
		self.lr = LogisticRegression(C=100.0, random_state=1)
		self.lr.fit(x, y)

	def predict(self, x, y):
		# todo what are these values telling us?
		y_pred_lr = self.lr.predict(x)
		y_pred_lr_prob = self.lr.predict_proba(x)
		error = (y != y_pred_lr).sum()
		print ("Misclassifications: " + str(error) + "/" + str(len(y_pred_lr)) + " = " + str(error/len(y_pred_lr) * 100) + "%")
