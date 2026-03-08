import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, f1_score


def compute_metrics_binary(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    auc = roc_auc_score(y_true, y_prob)
    return {
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
        "auc": auc,
    }


def compute_metrics_multiclass(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }


def per_class_metrics(y_true, y_pred, id2name):
    p, r, f1, s = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        labels=sorted(id2name.keys()),
        zero_division=0
    )

    rows = []
    for i in sorted(id2name.keys()):
        rows.append({
            "class_id": int(i),
            "class_name": id2name[i],
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(s[i]),
        })

    return pd.DataFrame(rows)


def summarize_fold_metrics(metrics_list):
    keys = list(metrics_list[0].keys())
    out = {}
    for k in keys:
        vals = np.array([m[k] for m in metrics_list], dtype=float)
        out[f"{k}_mean"] = float(vals.mean())
        out[f"{k}_std"] = float(vals.std(ddof=1))
    return out


def softmax_np(logits):
    e = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)