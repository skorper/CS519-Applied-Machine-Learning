import sys
import pandas as pd
import numpy as np

from perceptron import Perceptron
from adaline import Adaline
from sgd import Sgd
from ovr import Ovr

if len(sys.argv) != 3:
    print ("Incorrect usage. python main.py <algorithm> <dataset>")
    exit()

# Assuming this file is valid. I'm not really doing any checking in that regard.
df = pd.read_csv(sys.argv[2], header=None)

# set the '1' class
# todo remove hardcoding
percent_training = 70
dataset_size = len(df.index)
training_size = int(dataset_size * (percent_training/100))
dataset_columns = len(df.columns)
true_class = df.iloc[0, dataset_columns - 1] #just pick the first one for binary classification

# Load data (all columns)
x = df.iloc[0:dataset_size, 0:dataset_columns - 1].values # try to plot these values (first two dimensions) to test convergence and if data is linearly separable
y = df.iloc[0:dataset_size, dataset_columns - 1].values
y = np.where(y == true_class, 1, -1) 

# standardize (will converge extremely slowly otherwise)
# source: notes PDF

x_std = np.copy(x)
x_std[:, 0] = (x[:, 0] - x[:, 0].mean()) / x[:, 0].std()
x_std[:, 1] = (x[:, 1] - x[:, 1].mean()) / x[:, 1].std()

# Choose model
model = None

if sys.argv[1] == 'perceptron':
    model = Perceptron(_eta=0.01, _iter=20)
    x_std = x #running Perceptron on x_std gives weird results, so use original x
elif sys.argv[1] == 'adaline':
    model = Adaline(_eta=0.01, _iter=20)
elif sys.argv[1] == 'sgd':
    model = Sgd(_eta=0.01, _iter=20)
elif sys.argv[1] == 'ovr':
    model = Ovr(df.iloc[0:dataset_size, dataset_columns - 1].values, _eta=0.001, _iter=20)
else:
    print ("Incorrect algorithm. Options are perceptron, adaline, or sgd. ")
    exit()

# Train model

model.learn(x_std, y)

# Run predictions on model
# For the sake of this assignment, I'll be running the predictions
# on the rest of the data file.

#todo this is the last N values which are all the same class... should we shuffle?
x_tst = df.iloc[0:dataset_size, 0:dataset_columns - 1].values
y_tst = df.iloc[0:dataset_size, dataset_columns - 1].values

correct = 0
if sys.argv[1] == 'ovr':
    #This is different from binary classifier so handle separately
    for i in range(len(x_tst)):
        prediction = model.predict(x_tst[i])
        if prediction == y_tst[i]:
            correct += 1
else:
    for i in range(len(x_tst)):
        prediction = model.predict(x_tst[i])
        if (prediction == 1 and y_tst[i] == true_class) or (prediction == -1 and y_tst[i] != true_class):
            correct += 1

print (str(correct) + "/" + str(len(x_tst)) + " predictions were correct")
incorrect = (1 - (correct/len(x_tst))) * 100
print (str(incorrect) + "% incorrect predictions")    