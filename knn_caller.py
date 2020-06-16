import numpy as np 
from sklearn import datasets
from knn import KNN

def accuracy(y_true , y_pred):
    accuracy = np.sum(y_true == y_pred) /len(y_true)
    
    return accuracy

iris = datasets.load_iris()
X, y = iris.data, iris.target

k = 3
clf = KNN(k=k)
clf.fit(X, y)
predictions = clf.predict(X)
print(y)
print(predictions)
print("Accuracy = {0}".format(accuracy(y, predictions)))
