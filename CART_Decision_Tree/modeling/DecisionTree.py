from collections import Counter
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Optional, List


class DecisionTree(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth: int = 3, feature_labels: Optional[List[str]]=None, max_features: Optional[int]=None):
        self.max_depth = max_depth
        self.max_features = max_features
        self.feature_labels = feature_labels
        self.left: Optional['DecisionTree'] = None
        self.right: Optional['DecisionTree'] = None
        self.split_idx: Optional = None
        self.thresh: Optional =  None
        self.prediction: Optional[int] = None

    @staticmethod
    def entropy(y: np.ndarray[int]) -> float:
        length = len(y)
        if length == 0:
            return 0

        counts = np.bincount(y)
        probabilities = counts[counts > 0] / length
        return -np.sum(probabilities * np.log2(probabilities))

    @staticmethod
    def entropy_o1(counter: Counter[int], labels: np.ndarray, total_count: int) -> float:
        entropy = sum(
            count * np.log2(count / total_count) for count in (counter[label] for label in labels) if count > 0)
        return entropy

    @staticmethod
    def split(X: np.ndarray, y: np.ndarray[int], idx: int, thresh: float) -> tuple:
        left_mask = X[:, idx] < thresh
        right_mask = ~left_mask
        return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

    def fit(self, X: np.ndarray, y: np.ndarray[int]) -> 'DecisionTree':
        nb_samples, nb_features = X.shape
        labels = np.unique(y)

        if self.max_depth == 0 or self.entropy(y) == 0:
            self.prediction = Counter(y).most_common(1)[0][0]
            return self

        best_split = {'gain': 0}
        features_indices = np.random.choice(nb_features, size=min(self.max_features, nb_features),
                                            replace=False) if self.max_features else range(nb_features)

        for idx in features_indices:
            X_j = X[:, idx]
            sorted_indices = np.argsort(X_j)
            X_j_sorted, y_sorted = X_j[sorted_indices], y[sorted_indices]
            unique_values, indices = np.unique(X_j, return_index=True)
            y_unique = y[indices]
            if len(unique_values) < 2:
                continue

            y_entropy = self.entropy(y_sorted)
            total_sum = len(y_sorted)

            left_classes, right_classes = Counter(), Counter(y_sorted)
            counts_left, counts_right = 0, total_sum

            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                if i == 0:
                    left_classes = Counter(y_sorted[X_j_sorted < threshold])
                    right_classes = Counter(y_sorted[X_j_sorted >= threshold])
                    counts_left, counts_right = sum(left_classes.values()), sum(right_classes.values())
                else:
                    left_classes[y_unique[i]] += 1
                    counts_left += 1
                    right_classes[y_unique[i]] = right_classes[y_unique[i]] - 1
                    counts_right -= 1

                weighted_entropy = (-1 / total_sum) * (
                            self.entropy_o1(left_classes, labels, counts_left) + self.entropy_o1(right_classes, labels,
                                                                                                 counts_right))
                information_gain = y_entropy - weighted_entropy

                if information_gain > best_split['gain']:
                    best_split = {'gain': information_gain, 'threshold': threshold, 'feature': idx}

        if best_split['gain'] == 0:
            self.prediction = Counter(y).most_common(1)[0][0]
        else:
            self.split_idx = best_split['feature']
            self.thresh = best_split['threshold']
            X_left, y_left, X_right, y_right = self.split(X, y, self.split_idx, self.thresh)

            self.left = DecisionTree(self.max_depth - 1, self.feature_labels, self.max_features).fit(X_left, y_left)
            self.right = DecisionTree(self.max_depth - 1, self.feature_labels, self.max_features).fit(X_right, y_right)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.prediction is not None:
            return np.full(X.shape[0], self.prediction)

        predictions = np.empty(X.shape[0], dtype=int)
        feature_values = X[:, self.split_idx]
        left_mask = feature_values < self.thresh
        right_mask = ~left_mask

        predictions[left_mask] = self.left.predict(X[left_mask])
        predictions[right_mask] = self.right.predict(X[right_mask])

        return predictions
