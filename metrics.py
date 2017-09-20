
import numpy as np
from sklearn.metrics import accuracy_score

def majority_voting_score(X, y, estimators, classes):
    voting_matrix = np.zeros((X.shape[0], len(classes)))
    for estimator in estimators:
        predictions = estimator.predict(X)
        for i in range(X.shape[0]):
            voting_matrix[i, predictions[i]] += 1
    voting_score = classes.take(np.argmax(voting_matrix, axis=1))
    return accuracy_score(y, voting_score)

