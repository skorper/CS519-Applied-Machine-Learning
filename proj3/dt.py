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

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion = 'gini', max_depth = 4, random_state = 1)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
accuracy = (y_test == y_pred_tree).sum()/len(y_test)
print ('accuracy %.2f' %accuracy)
