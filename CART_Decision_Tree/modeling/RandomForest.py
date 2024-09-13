from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from typing import Optional, List

class BaggedTrees(BaseEstimator, ClassifierMixin):
    def __init__(self, Tree, params: Optional[dict] = None, n: Optional[int] = 200):
        if params is None:
            params = {}
        self.params: Optional[dict] = params
        self.n: Optional[int] = n
        self.Tree = Tree
        self.decision_trees = [
            Tree(**self.params) for _ in range(self.n)
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaggedTrees':
        nb_samples = X.shape[0]
        for decisionTree in self.decision_trees:
            indices = np.random.choice(nb_samples, size=nb_samples, replace=True)
            decisionTree.fit(X[indices], y[indices])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        decisions = np.array([decisionTree.predict(X) for decisionTree in self.decision_trees])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=decisions)


class RandomForest(BaggedTrees):
    def __init__(self, Tree, params: Optional[dict] = None, n: Optional[int] = 200, m: Optional[int] = 1):
        if params is None:
            params = {}
        self.m: Optional[int] = m
        self.n: Optional[int] = n
        self.Tree = Tree
        # Do not modify params directly, keep it immutable
        super().__init__(Tree, params, n)

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Set the max_features for each tree in the fit method, where changes won't affect cloning
        for decisionTree in self.decision_trees:
            # Set max_features separately for each tree (if it's implemented in the tree)
            if hasattr(decisionTree, 'max_features'):
                setattr(decisionTree, 'max_features', self.m)
        return super().fit(X, y)
