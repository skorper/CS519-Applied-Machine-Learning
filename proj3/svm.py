################################
#                              #
# svm.py Simple Vector Machine #
# Author: Stepheny Perez       #
#                              #
# This class uses the scikit   #
# python library to create a   #
# SVM classifier.              #
#                              #
################################

from sklearn.svm import SVC

svm = SVC(kernel=rbf, random_state=1, gamma=0.2, C=1.0)
svm.fit(X_train_std, y_train)
