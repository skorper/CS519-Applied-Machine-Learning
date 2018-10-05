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

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=100.0, random_state=1)
lr.fit(X_train_std, y_train)
y_pred_lr = lr.predict(X_test_std)
y_pred_lr_prob = lr.predict_proba(X_test_std)
print(y_pred_lr_prob.shape)
print(y_pred_lr_prob)
