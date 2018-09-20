import sys
import pandas as pd
import numpy as np

from perceptron import Perceptron
from adaline import Adaline
from sgd import Sgd

if len(sys.argv) != 3:
    print ("Incorrect usage. python main.py <algorithm> <dataset>")
    exit()

# Assuming this file is valid. I'm not really doing any checking in that regard.
df = pd.read_csv(sys.argv[2], header=None)

# set the '1' class
true_class = "iris-setosa"

# Load data
x = df.iloc[0:100, [0,2]].values # try to plot these values (first two dimensions) to test convergence and if data is linearly separable
y = df.iloc[0:100, 4].values
y = np.where(y == true_class, 1, -1) # todo remove hardcoding

# standardize (will converge extremely slowly otherwise)
# source: notes PDF

x_std = np.copy(x)
x_std[:, 0] = (x[:, 0] - x[:, 0].mean()) / x[:, 0].std()
x_std[:, 1] = (x[:, 1] - x[:, 1].mean()) / x[:, 1].std()

# Choose model
model = None

if sys.argv[1] == 'perceptron':
    model = Perceptron(_eta=0.01, _iter=20)
    x_std = x #running Perceptron on x_std gives weird results, so resort to original x
elif sys.argv[1] == 'adaline':
    model = Adaline(_eta=0.01, _iter=20)
elif sys.argv[1] == 'sgd':
    model = Sgd(_eta=0.01, _iter=20)
else:
    print ("Incorrect algorithm. Options are perceptron, adaline, or sgd. ")
    exit()

# Train model

model.learn(x_std, y)

# Run predictions on model
# For the sake of this assignment, I'll be running the predictions
# on the rest of the data file.

#todo remove hardcoding
#todo this is the last N values which are all the same class... should we shuffle?
x_tst = df.iloc[100:148, [0,2]].values
y_tst = df.iloc[100:148, 4].values

correct = 0

for i in range(len(x_tst)):
    prediction = model.predict(x_tst[i])
    if (prediction == 1 and y_tst[i] == true_class) or (prediction == -1 and y_tst[i] != true_class):
        correct += 1

print (str(correct) + "/" + str(len(x_tst)) + " predictions were correct")
incorrect = (1 - (correct/len(x_tst))) * 100
print (str(incorrect) + "% incorrect predictions")