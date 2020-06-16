import numpy as np
from collections import Counter
class KNN:

    def __init__(self, k=3):
        self.k = k


    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    

    def predict(self, X):
        predicted_labels = [self._predictLabel(x) for x in X]
        return predicted_labels


    def _getDistance(self, x, y):
        return np.sqrt(np.sum((x-y)**2))

    def _predictLabel(self, x):
        distances = [self._getDistance(x, y) for y in self.X_train]
        indices = np.argsort(distances)[:self.k]
        labels = [self.y_train[i] for i in indices]
        majority = Counter(labels).most_common(1)
        occurence_tuple = majority[0]
        print("max {0}".format(occurence_tuple[0]))
        return occurence_tuple[0]


