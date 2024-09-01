from DecisionTree import DecisionTree
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from typing import Optional, List

class BaggedTrees(BaseEstimator, ClassifierMixin):
    # Inherits form BaseEstimator, ClassifierMixin So we can use sklearn packages
    def __init__(self, params: Optional[dict]=None, n: Optional[int]=200):
        if params is None:
            params = {}
        self.params : Optional[dict] = params
        self.n : Optional[int] = n
        self.decision_trees: List['DecisionTree'] = [
             DecisionTree(**self.params) for _ in range(self.n)
        ]

    def fit(self, X: np.ndarray, y: np.ndarray[int]) -> 'BaggedTrees':
        nb_samples = X.shape[0]
        for decisionTree in self.decision_trees:
            indices = np.random.choice(nb_samples, size=nb_samples, replace=True)
            decisionTree.fit(X[indices], y[indices])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        decisions = np.array([decisionTree.predict(X) for decisionTree in self.decision_trees])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=decisions)


class RandomForest(BaggedTrees):
    def __init__(self, params: Optional[dict]=None, n: Optional[int]=200, m: Optional[int]=1):
        if params is None:
            params = {}
        params["max_features"] = m
        self.params: Optional[dict] = params
        # We add these here for the BaseEstimator
        self.m : Optional[int] = m
        self.n : Optional[int]= n
        super().__init__(params, n)


class BoostedRandomForest(RandomForest):

    def fit(self, X, y):
        # TODO
        pass

    def predict(self, X):
        # TODO
        pass
