
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils import check_array

def _voting_probability(estimators, X, n_classes):
    n_samples = X.shape[0]
    proba = np.zeros((n_samples, n_classes))
    for estimator in estimators:
        predictions = estimator.predict(X)
        for i in range(n_samples):
            proba[i, predictions[i]] += 1
    return proba

def _predict_probability(estimators, X, n_classes):
    X = check_array(X, accept_sparse=['csr', 'csc'])
    all_proba = _voting_probability(estimators, X, n_classes)
    return all_proba

def _predict(estimators, X, classes):
    predicted_probability = _predict_probability(estimators, X, len(classes))
    return classes.take((np.argmax(predicted_probability, axis=1)), axis=0)

def majority_voting_score(X, y, estimators, classes):
    return accuracy_score(y, _predict(estimators, X, classes))