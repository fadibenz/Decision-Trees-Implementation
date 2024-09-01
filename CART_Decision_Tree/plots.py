import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix, roc_curve, auc


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