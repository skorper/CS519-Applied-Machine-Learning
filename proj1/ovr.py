import numpy as np

from sgd import Sgd

class Ovr(object):
    def __init__(self, _original_y, _eta=0.1, _iter=10):
        self.eta = _eta
        self.iter = _iter
        self.original_y = _original_y

    def learn(self, x, y):
        # Get each class (extract distinct values from y)
        self.classes = []
        for class_label in self.original_y:
            if class_label not in self.classes:
                self.classes.append(class_label)

        self.classifiers = []

        # Create a new Sgd classifier for each class
        for true_class in self.classes:
            print ("Training Sdg classifier for +1 class = " + str(true_class))
            # set +1 / -1 depending on which classifier we're building
            y = np.where(self.original_y == true_class, 1, -1) 
            # build each classifier and train with above y values
            self.classifiers.append(Sgd(self.eta, self.iter))
            self.classifiers[-1].learn(x, y)
        return self

    def predict(self, x):
        count = [0] * len(self.classifiers)
        for i in range(len(self.classifiers)):
            count[i] += self.classifiers[i].predict(x)        
        max_index = count.index(max(count))
        return self.original_y[max_index]