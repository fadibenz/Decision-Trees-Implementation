import pickle
from DecisionTree import DecisionTreeCategorical
from RandomForest import RandomForest
import numpy as np

if __name__ == "__main__":
    X = np.genfromtxt('data/interim/titanic_training_V3.csv', delimiter=",", dtype=float)[1:, ]
    y = np.genfromtxt('data/processed/titanic_training_labels.csv', delimiter=",", dtype=int)[1:]

    params = {
        'max_depth': 21,
        'cat_cols': [0, 1, 6]
    }

    model = RandomForest(DecisionTreeCategorical, n=300, m=3, params=params)
    model.fit(X, y)
    with open('models/model_v2/model_v2.pkl', 'wb') as f:
        pickle.dump(model, f)


