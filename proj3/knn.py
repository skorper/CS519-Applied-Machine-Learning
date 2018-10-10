################################# 
#                               #
# knn.py K-Nearest Neighbor     #
# Author: Stepheny Perez        #
#                               #
# This class uses the scikit    #
# python library to create a    #
# k-nearest-neighbor classifier #
#                               # 
#################################

# Note: In order to create this class I used the following references:
#	- 03_04_knn.pdf (from Canvas)
#	- http://scikit-learn.org/stable/modules/neighbors.html

from sklearn.neighbors import KNeighborsClassifier

# knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
# knn.fit(X_train_std, t_train)
# y_pred_knn = knn.predict(X_test_std)

class KNN(object):

	def __init__(self, _n_neighbors=5, _p=2, _metric='minkowski'):
		self.n_neighbors = _n_neighbors
		self.p = _p
		self.metric = _metric

	def learn(self, x, y):
		self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors, p=self.p, metric=self.metric)
		self.knn.fit(x, y)

	def predict(self, x, y):
		y_pred = self.knn.predict(x)
		error = (y != y_pred).sum()
		print ("Misclassifications: " + str(error) + "/" + str(len(y_pred)) + " = " + str(error/len(y_pred) * 100) + "%")
