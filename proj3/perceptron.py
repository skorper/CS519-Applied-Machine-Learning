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

# TODO create a class def

from sklearn.linear_model import Perceptron

# load data
# TODO

# train model
ppn = Perceptron(max_iter=40, eta0=0.1, random_state=1)
ppn.fit(X_tr, y_tr)

# predict/print error
y_pred = ppn.predict(X_ts)
error = (y_ts != y_pred).sum()
print ("Misclassifications: " + error)
