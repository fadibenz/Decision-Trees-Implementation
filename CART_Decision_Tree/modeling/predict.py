import pandas as pd
import pickle
import numpy as np


def results_to_csv(y_test):
    y_test = y_test.astype(int)
    df = pd.DataFrame({'Category': y_test})
    df.index += 1 # Ensures that the index starts at 1
    df.to_csv('models/model_v1/predictions.csv', index_label='Id')


if __name__ == "__main__":
    with open('models/model_v1/model_v1.pkl', 'rb') as f:
        model = pickle.load(f)
    test_data = np.genfromtxt('data/processed/titanic_test_cleaned.csv', delimiter=',', dtype=float)[1:, :]
    predictions = model.predict(test_data)
    results_to_csv(predictions)
