import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

# Actual answers
# 0 = Not Spam
# 1 = Spam

y_actual = np.array([
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1
])

probabilities = np.array([ 0.05,
                           0.20,
                           0.40,
                           0.45,
                           0.60,
                           0.55,
                           0.65,
                           0.75,
                           0.85,
                           0.95])

threshold = 0.5

thresholds = [0.3, 0.5, 0.7, 0.9]


for threshold in thresholds:

    # Convert probabilities into 0/1 predictions
    y_pred = (probabilities >= threshold).astype(int)

    print(f"\nThreshold: {threshold}")
    print("Predictions:", y_pred)

    print(
        "Accuracy:",
        accuracy_score(y_actual, y_pred)
    )

    print(
        "Precision:",
        precision_score(y_actual, y_pred)
    )

    print(
        "Recall:",
        recall_score(y_actual, y_pred)
    )

    print(
        "F1:",
        f1_score(y_actual, y_pred)
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_actual, y_pred))


# ROC-AUC uses the probabilities,
# not the thresholded 0/1 predictions.
auc = roc_auc_score(y_actual, probabilities)

print("\nROC-AUC:", auc)