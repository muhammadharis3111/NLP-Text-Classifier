import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


def evaluate(model, X, y_true):
    preds = model.predict(X)
    acc = accuracy_score(y_true, preds)
    correct = int(np.sum(preds == y_true))
    incorrect = len(y_true) - correct
    cm = confusion_matrix(y_true, preds)
    return {"accuracy": acc, "correct": correct, "incorrect": incorrect,
            "predictions": preds, "confusion_matrix": cm}
