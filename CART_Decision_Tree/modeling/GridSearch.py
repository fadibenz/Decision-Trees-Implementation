import numpy as np
from DecisionTree import DecisionTreeCategorical
from RandomForest import RandomForest
from sklearn.model_selection import GridSearchCV

X = np.genfromtxt('data/interim/titanic_training_V3.csv', delimiter=",", dtype=float)[1:, ]
y = np.genfromtxt('data/processed/titanic_training_labels.csv', delimiter=",", dtype=int)[1:]

param_grid = {
    'n': [100, 200, 300, 400],  # Number of trees in the forest
    'params': [{'max_depth': 14, 'cat_cols':[0, 1, 6]},
               {'max_depth': 12, 'cat_cols':[0, 1, 6]},
               {'max_depth': 10, 'cat_cols':[0, 1, 6] },
               {'max_depth': 8, 'cat_cols':[0, 1, 6] },
               {'max_depth': 3, 'cat_cols':[0, 1, 6] }]
,  # Different depths to try
    'm': [2, 3]  # max_features for each tree
}

print('\n\n Decision tree categorical')
model = RandomForest(DecisionTreeCategorical)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_search.fit(X, y)
print("Best hyperparameter:",grid_search.best_params_)