import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix, roc_curve, auc
from modeling.DecisionTree import DecisionTreeCategorical
from modeling.RandomForest import RandomForest
from sklearn.model_selection import cross_val_score, train_test_split

def plot_learning_curve(estimator, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)):
    train_sizes, train_scores, valid_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                                                             train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)

    plt.figure(figsize=(10, 7))
    plt.plot(train_sizes, train_scores_mean, 'o-', color='green', label='Training score')
    plt.plot(train_sizes, valid_scores_mean, 'o-', color='red', label='Cross-validation score')
    plt.title('Learning Curve')
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    plt.legend(loc="best")
    plt.show()


def plot_learning_curve_depth(X, y, cv=3, n_jobs=-1):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=0)

    max_depth_range = np.arange(18, 30)  # Example range for max_depth
    train_scores = []
    val_scores = []

    # Loop over different values of max_depth
    for max_depth in max_depth_range:
        params = {
            'max_depth': max_depth,
            'cat_cols': [0, 1, 6]  # Use your relevant columns
        }

        # Initialize your RandomForest model with varying max_depth
        model = RandomForest(DecisionTreeCategorical, n=300, m=3, params=params)

        # Compute training score (using cross-validation for consistency)
        train_score = np.mean(cross_val_score(model, X_train, y_train, cv=cv, n_jobs=n_jobs))
        train_scores.append(train_score)

        # Compute validation score
        val_score = np.mean(cross_val_score(model, X_val, y_val, cv=cv, n_jobs=n_jobs))
        val_scores.append(val_score)

    # Convert scores to numpy arrays for easier plotting
    train_scores = np.array(train_scores)
    val_scores = np.array(val_scores)

    # Plot the validation and training scores
    plt.plot(max_depth_range, train_scores, label="Training score", color="r")
    plt.plot(max_depth_range, val_scores, label="Validation score", color="g")

    plt.title("Validation Curve for RandomForest max_depth")
    plt.xlabel("max_depth")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.show()
    plt.savefig('validation_depth.png')

def plot_feature_importance(model, feature_names):
    feature_importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)

    plt.figure(figsize=(10, 7))
    feature_importance.plot(kind='bar', color='teal')
    plt.title('Feature Importance')
    plt.ylabel('Importance')
    plt.xlabel('Features')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, labels=None, normalize=False, cmap='Blues'):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true' if normalize else None)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap, xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


def plot_roc_curve(y_true, y_pred_proba, pos_label=1):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    X = np.genfromtxt('data/interim/titanic_training_V3.csv', delimiter=",", dtype=float)[1:, ]
    y = np.genfromtxt('data/processed/titanic_training_labels.csv', delimiter=",", dtype=int)[1:]

    plot_learning_curve_depth(X, y)
