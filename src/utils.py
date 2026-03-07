import os


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def pretty_view(df_res):
    cols = [
        "model",
        "accuracy_mean", "accuracy_std",
        "precision_mean", "precision_std",
        "recall_mean", "recall_std",
        "f1_mean", "f1_std",
        "auc_mean", "auc_std"
    ]
    existing_cols = [c for c in cols if c in df_res.columns]
    return df_res[existing_cols]