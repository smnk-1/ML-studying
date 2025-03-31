import numpy as np

def accuracy_score(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def confusion_matrix(y_true, y_pred):
    classes = np.unique(np.concatenate((y_true, y_pred)))
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for i, a in enumerate(classes):
        for j, p in enumerate(classes):
            matrix[i, j] = np.sum((y_true == a) & (y_pred == p))
    return matrix

def precision_score(y_true, y_pred):
    classes = np.unique(np.concatenate((y_true, y_pred)))
    prec = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        prec.append(tp / (tp + fp) if tp + fp != 0 else 0)
    return np.mean(prec)

def recall_score(y_true, y_pred):
    classes = np.unique(np.concatenate((y_true, y_pred)))
    rec = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fn = np.sum((y_pred != c) & (y_true == c))
        rec.append(tp / (tp + fn) if tp + fn != 0 else 0)
    return np.mean(rec)

def f1_score(y_true, y_pred):
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return 2 * p * r / (p + r) if p + r != 0 else 0
