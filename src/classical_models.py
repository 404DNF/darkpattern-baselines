import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from src.config import SEED, N_SPLITS
from src.metrics import compute_metrics_binary, summarize_fold_metrics

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except Exception as e:
    print(f"⚠️ LightGBM import skipped: {e}")
    LIGHTGBM_AVAILABLE = False


def run_classical_models(df):
    X = df["text"].values
    y = df["label"].values

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    bow = CountVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
        dtype=np.float32,
    )

    models = {
        "LogReg": LogisticRegression(C=4.95, max_iter=2000, solver="lbfgs", random_state=SEED),
        "SVM(rbf)": SVC(kernel="rbf", C=4.35, probability=True, random_state=SEED),
        "RandomForest": RandomForestClassifier(
            max_depth=32,
            n_estimators=641,
            min_samples_split=18,
            min_samples_leaf=1,
            bootstrap=False,
            n_jobs=-1,
            random_state=SEED,
        ),
    }

    if LIGHTGBM_AVAILABLE:
        models["LightGBM"] = LGBMClassifier(
            reg_alpha=3.04e-8,
            reg_lambda=0.806,
            num_leaves=89,
            colsample_bytree=0.466,
            subsample=0.887,
            subsample_freq=6,
            min_child_samples=5,
            objective="binary",
            n_estimators=1000,
            learning_rate=0.05,
            random_state=SEED,
            n_jobs=-1,
            verbosity=-1,
            force_row_wise=True,
        )

    all_rows = []
    for name, clf in models.items():
        print(f"Running Classical Model: {name}")
        fold_metrics = []
        for fold, (tr, te) in enumerate(skf.split(X, y), 1):
            print(f"   - Fold {fold}/{N_SPLITS}")
            pipe = Pipeline([("bow", bow), ("clf", clf)])
            pipe.fit(X[tr], y[tr])

            y_prob = pipe.predict_proba(X[te])[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            fold_metrics.append(compute_metrics_binary(y[te], y_pred, y_prob))

        all_rows.append({"model": name, **summarize_fold_metrics(fold_metrics)})

    return pd.DataFrame(all_rows).sort_values("accuracy_mean", ascending=False)