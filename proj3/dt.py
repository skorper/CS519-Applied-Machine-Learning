##############################
#                            #
# dt.py Decision Tree        #
# Author: Stepheny Perez     #
#                            #
# This class uses the scikit #
# python library to create a #
# decision tree classifier.  #
#                            #
##############################

# Note: In order to create this class I used the following references:
#	- 03_03_decision_tree.pdf (from Canvas)
#	- http://scikit-learn.org/stable/modules/tree.html

from sklearn.tree import DecisionTreeClassifier

# tree = DecisionTreeClassifier(criterion = 'gini', max_depth = 4, random_state = 1)
# tree.fit(X_train, y_train)
# y_pred_tree = tree.predict(X_test)
# accuracy = (y_test == y_pred_tree).sum()/len(y_test)
# print ('accuracy %.2f' %accuracy)

class DT(object):

	def __init__(self, _criterion='gini', _max_depth=4):
		self.criterion = _criterion
		self.max_depth = _max_depth

	def learn(self, x, y):
		self.tree = DecisionTreeClassifier(criterion = self.criterion, max_depth = self.max_depth, random_state = 1)
		self.tree.fit(x, y)

	def predict(self, x, y):
		y_pred = self.tree.predict(x)
		error = (y != y_pred).sum()
		print ("Misclassifications: " + str(error) + "/" + str(len(y_pred)) + " = " + str(error/len(y_pred) * 100) + "%")
