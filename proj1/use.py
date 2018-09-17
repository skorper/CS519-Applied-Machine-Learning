import pandas as pd
import numpy as np
from perceptron import Perceptron

df = pd.read_csv('iris.data', header=None)

# right now our perceptron can only do binary class labels (and iris has 3 labels)

x = df.iloc[0:100, [0,2]].values # try to plot these values (first two dimensions) to test convergence and if data is linearly separable
y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", 1, -1) # todo change this back to lower case

model = Perceptron(_eta=0.1, _iter=10)
model.learn(x, y)
#print error in different iterations

#how to predict?

