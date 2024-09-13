from collections import Counter
import numpy as np
import scipy
import pandas as pd


def preprocess(data, fill_mode=True, min_freq=10, onehot_cols=None):
    """
    :param data:
    :param fill_mode: Whether to fill missing values with mode of feature
    :param min_freq: The minimum frequency required for a feature to be considered
    :param onehot_cols: columns that contain categorical features

    :return: Preprocessed data and feature names
    """

    # Temporarily assign -1 to missing data
    if onehot_cols is None:
        onehot_cols = []

    data[data == b""] = "-1"

    # Hash the columns (used for handling strings)
    onehot_encoding = []
    onehot_features = []
    for col in onehot_cols:
        counter = Counter(
            data[:, col]
        )  # Here we are getting all the possible values for this categorical feature 'col'
        for term in counter.most_common():
            if term[0] == b"-1":
                continue
            if (
                term[-1] <= min_freq
            ):  # Check how many times did this attribute appear, if it's less than ten
                break
            onehot_features.append(term[0])
            onehot_encoding.append((data[:, col] == term[0]).astype(float))
        data[:, col] = "0"
    onehot_encoding = np.array(onehot_encoding).T
    data = np.hstack([np.array(data, dtype=float), np.array(onehot_encoding)])

    # Replace missing data with the mode value. We use the mode instead of
    # the mean or median because this makes more sense for categorical
    # features such as gender or cabin type, which are not ordered.

    if fill_mode:
        data_with_nan = np.where(data == -1, np.nan, data)
        modes = scipy.stats.mode(data_with_nan, nan_policy="omit")[0]
        indices = np.where(data == -1)
        for i, j in zip(indices[0], indices[1]):
            data[i, j] = modes[j]
    return data, onehot_features





if __name__ == "__main__":
    path_train = 'data/raw/titanic_training.csv'
    data = np.genfromtxt(path_train, delimiter=',', dtype=None)
    path_test = 'data/raw/titanic_testing_data.csv'
    test_data = np.genfromtxt(path_test, delimiter=',', dtype=None)
    y = data[1:, 0]  # label = survived
    class_names = ["Died", "Survived"]

    labeled_idx = np.where(y != b'')[0]
    y = np.array(y[labeled_idx], dtype=float).astype(int)
    X, onehot_features = preprocess(data[1:, 1:], onehot_cols=[0,1, 5, 7, 8])

    X = X[labeled_idx, :]
    Z, _ = preprocess(test_data[1:, :], onehot_cols=[0, 1, 5, 7, 8])

    features = list(data[0, 1:]) + onehot_features
    df_train = pd.DataFrame(X, columns=features)
    df_test = pd.DataFrame(Z, columns=features)
    df_labels = pd.DataFrame(y, columns=['label'])

    df_train.to_csv('data/processed/titanic_training_cleaned.csv', index=False)
    df_labels.to_csv('data/processed/titanic_training_labels.csv', index=False)
    df_test.to_csv('data/processed/titanic_test_cleaned.csv', index=False)
