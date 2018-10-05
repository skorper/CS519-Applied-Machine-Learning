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

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train_std, t_train)
y_pred_knn = knn.predict(X_test_std)

