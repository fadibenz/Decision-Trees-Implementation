import pickle
from DecisionTree import DecisionTree
import numpy as np

if __name__ == "__main__":
    X = np.genfromtxt('data/processed/titanic_training_cleaned.csv', delimiter=",", dtype=float)[1:, :]
    y = np.genfromtxt('data/processed/titanic_training_labels.csv', delimiter=",", dtype=int)[1:]
    features = list(X[0])

    model = DecisionTree()
    model.fit(X, y)
    with open('models/model_v1/model_v1.pkl', 'wb') as f:
        pickle.dump(model, f)
