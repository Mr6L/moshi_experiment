"""手写分类指标；混淆矩阵的行是真实类别，列是预测类别。"""

import numpy as np


def classification_metrics(y_true, y_pred, n_classes=3):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape or y_true.ndim != 1 or len(y_true) == 0:
        raise ValueError("真实标签和预测标签必须是等长非空一维数组")
    if ((y_true < 0) | (y_true >= n_classes) | (y_pred < 0) | (y_pred >= n_classes)).any():
        raise ValueError("分类指标的内部标签超出类别范围")
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    tp = np.diag(matrix).astype(float)
    predicted, support = matrix.sum(axis=0), matrix.sum(axis=1)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    return {
        "accuracy": float(tp.sum() / len(y_true)),
        "macro_f1": float(f1.mean()),
        "errors": int(len(y_true) - tp.sum()),
        "confusion_matrix": matrix.tolist(),
        "per_class": [
            {"label": i + 1, "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])}
            for i in range(n_classes)
        ],
    }
