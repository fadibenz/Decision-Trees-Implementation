from collections import Counter
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Optional, List
from scipy import stats
from itertools import combinations

class DecisionTreeCategorical(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth: int = 3, feature_labels: Optional[List[str]]=None, max_features: Optional[int]=None,
                 cat_cols=None):
        if cat_cols is None:
            cat_cols = []
        self.cat_cols = cat_cols
        self.max_depth = max_depth
        self.max_features = max_features
        self.feature_labels = feature_labels
        self.left: Optional = None
        self.right: Optional = None
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
    def get_subsets_combinations(arr):
        all_subsets = []
        for r in range(1, len(arr)):
            subsets = combinations(arr, r)
            all_subsets.extend(subsets)
        return [np.array(subset) for subset in all_subsets]

    def fill_infer_cat(self, X, idx):
        indices = np.where(X == -1)
        X_with_nan = np.where(X == -1, np.nan, X)
        non_nan_values = X_with_nan[~np.isnan(X_with_nan)]

        if len(non_nan_values) == 0:
            return X, True

        if idx in self.cat_cols:
            fill_with = stats.mode(non_nan_values)[0]
        else:
            fill_with = np.mean(non_nan_values)

        X_with_nan[indices] = fill_with
        return X_with_nan, False

    def information_gain(self, X, y, subset):
        total_entropy = self.entropy(y)

        left_mask = np.isin(X, subset[0])
        right_mask = ~left_mask

        y_right = y[right_mask]
        y_left = y[left_mask]

        if not len(y_right) or not len(y_left):
            return 0, left_mask, right_mask

        weighted_entropy = (len(y_right) * self.entropy(y_right) + len(y_left) * self.entropy(y_left)) / len(y)
        return total_entropy - weighted_entropy, left_mask, right_mask

    def fit(self, X: np.ndarray, y: np.ndarray[int]) -> 'DecisionTreeCategorical':
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
            X_j, stop = self.fill_infer_cat(X_j, idx)
            # When all values are missing I'm skipping the node
            if stop:
                continue
            sorted_indices = np.argsort(X_j)
            X_j_sorted, y_sorted = X_j[sorted_indices], y[sorted_indices]
            unique_values, indices = np.unique(X_j, return_index=True)
            y_unique = y[indices]

            if len(unique_values) < 2:
                continue

            if idx in self.cat_cols:
                subsets_combinations = self.get_subsets_combinations(unique_values)
                n = len(subsets_combinations)
                for index, combination in enumerate(subsets_combinations):
                    if index == n / 2:
                        break
                    split = (combination, subsets_combinations[n - index - 1])
                    information_gain_2, left_mask, right_mask = self.information_gain(X_j, y, split)
                    if information_gain_2 > best_split['gain']:
                        best_split = {'gain': information_gain_2, 'threshold': split, 'feature': idx,
                                      'left_mask': left_mask, 'right_mask': right_mask}
            else:
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
                    left_mask = X_j < threshold
                    right_mask = ~left_mask

                    if information_gain > best_split['gain']:
                        best_split = {'gain': information_gain, 'threshold': threshold, 'feature': idx,
                                      'left_mask': left_mask, 'right_mask': right_mask}

        if best_split['gain'] == 0:
            self.prediction = Counter(y).most_common(1)[0][0]
        else:
            self.split_idx = best_split['feature']
            self.thresh = best_split['threshold']
            left_mask = best_split['left_mask']
            right_mask = best_split['right_mask']
            X_left, y_left, X_right, y_right = X[left_mask], y[left_mask], X[right_mask], y[right_mask]
            self.left = DecisionTreeCategorical(self.max_depth - 1, self.feature_labels, self.max_features).fit(X_left,
                                                                                                                y_left)
            self.right = DecisionTreeCategorical(self.max_depth - 1, self.feature_labels, self.max_features).fit(
                X_right, y_right)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:

        if not hasattr(self, '_imputed'):
            for feature in range(X.shape[1]):
                X[:, feature], _ = self.fill_infer_cat(X[:, feature], feature)
            self._imputed = True

        if self.prediction is not None:
            return np.full(X.shape[0], self.prediction)

        if isinstance(self.thresh, tuple):  # Categorical split
            left_mask = np.isin(X[:, self.split_idx], self.thresh[0])
        else:
            left_mask = X[:, self.split_idx] < self.thresh

        predictions = np.empty(X.shape[0], dtype=int)
        predictions[left_mask] = self.left.predict(X[left_mask])
        predictions[~left_mask] = self.right.predict(X[~left_mask])

        return predictions
