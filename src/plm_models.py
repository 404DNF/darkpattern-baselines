import os
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed
)

from src.config import SEED, N_SPLITS, MAX_LENGTH
from src.metrics import compute_metrics_binary, summarize_fold_metrics, softmax_np


device = "cuda" if torch.cuda.is_available() else "cpu"

PLM_SPECS = {
    "BERT-base":    {"ckpt": "bert-base-uncased",   "bs": 16, "lr": 4e-5, "dropout": 0.1, "epochs": 5},
    "BERT-large":   {"ckpt": "bert-large-uncased",  "bs": 32, "lr": 3e-5, "dropout": 0.1, "epochs": 5},
    "RoBERTa-base": {"ckpt": "roberta-base",        "bs": 32, "lr": 3e-5, "dropout": 0.1, "epochs": 5},
    "RoBERTa-large":{"ckpt": "roberta-large",       "bs": 16, "lr": 3e-5, "dropout": 0.1, "epochs": 5},
    "ALBERT-base":  {"ckpt": "albert-base-v2",      "bs": 16, "lr": 3e-5, "dropout": 0.1, "epochs": 5},
    "ALBERT-large": {"ckpt": "albert-large-v2",     "bs": 16, "lr": 5e-5, "dropout": 0.1, "epochs": 5},
    "XLNet-base":   {"ckpt": "xlnet-base-cased",    "bs": 8,  "lr": 2e-5, "dropout": 0.1, "epochs": 5},
    "XLNet-large":  {"ckpt": "xlnet-large-cased",   "bs": 4,  "lr": 4e-5, "dropout": 0.1, "epochs": 5},
}


def trainer_compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = softmax_np(logits)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return compute_metrics_binary(labels, preds, probs)


def set_dropout_in_config(cfg, dropout):
    for attr in ["hidden_dropout_prob", "attention_probs_dropout_prob", "classifier_dropout", "dropout"]:
        if hasattr(cfg, attr) and getattr(cfg, attr) is not None:
            setattr(cfg, attr, dropout)
    return cfg


def _load_model_safe(ckpt, cfg, model_key):
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            ckpt,
            config=cfg,
            use_safetensors=True
        ).to(device)
        return model
    except Exception as e:
        print(f"⚠️ {model_key} skipped: cannot load with safetensors. ({type(e).__name__}) {e}")
        return None


def run_plm_5fold(df, model_key, out_root):
    set_seed(SEED)

    X = df["text"].tolist()
    y = df["label"].astype(int).tolist()
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    spec = PLM_SPECS[model_key]
    ckpt, bs, lr, dr, epochs = (
        spec["ckpt"], spec["bs"], spec["lr"], spec["dropout"], spec["epochs"]
    )

    tokenizer = AutoTokenizer.from_pretrained(ckpt, use_fast=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    fold_metrics = []

    for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        print(f"   - Fold {fold}/{N_SPLITS}")

        train_df = pd.DataFrame({
            "text": [X[i] for i in tr],
            "label": [y[i] for i in tr]
        })
        test_df = pd.DataFrame({
            "text": [X[i] for i in te],
            "label": [y[i] for i in te]
        })

        train_ds = Dataset.from_pandas(train_df)
        eval_ds = Dataset.from_pandas(test_df)

        def tok(batch):
            return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

        train_ds = train_ds.map(tok, batched=True, remove_columns=["text"])
        eval_ds = eval_ds.map(tok, batched=True, remove_columns=["text"])

        cfg = AutoConfig.from_pretrained(ckpt, num_labels=2)
        cfg = set_dropout_in_config(cfg, dr)

        model = _load_model_safe(ckpt, cfg, model_key)
        if model is None:
            return None

        args = TrainingArguments(
            output_dir=f"{out_root}/{model_key.replace('/', '_')}/fold{fold}",
            per_device_train_batch_size=bs,
            per_device_eval_batch_size=bs,
            learning_rate=lr,
            num_train_epochs=epochs,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="no",
            load_best_model_at_end=False,
            logging_strategy="epoch",
            report_to="none",
            fp16=(device == "cuda"),
            seed=SEED
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=data_collator,
            compute_metrics=trainer_compute_metrics
        )

        trainer.train()
        eval_out = trainer.evaluate()

        fold_metrics.append({
            "accuracy": eval_out["eval_accuracy"],
            "precision": eval_out["eval_precision"],
            "recall": eval_out["eval_recall"],
            "f1": eval_out["eval_f1"],
            "auc": eval_out["eval_auc"],
        })

        del model, trainer
        if device == "cuda":
            torch.cuda.empty_cache()

    return summarize_fold_metrics(fold_metrics)


def run_plm_models(df, model_keys, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    rows = []
    partial_path = os.path.join(save_dir, "plm_results_partial.csv")

    for k in model_keys:
        print(f"\n Running {k}")
        summ = run_plm_5fold(df, k, out_root=save_dir)

        if summ is None:
            print(f"⏭ skipped: {k}")
            continue

        row = {"model": k, **summ}
        rows.append(row)

        pd.DataFrame(rows).to_csv(partial_path, index=False)
        print("saved partial:", partial_path)

    if len(rows) == 0:
        return pd.DataFrame(columns=[
            "model",
            "accuracy_mean", "accuracy_std",
            "precision_mean", "precision_std",
            "recall_mean", "recall_std",
            "f1_mean", "f1_std",
            "auc_mean", "auc_std"
        ])

    return pd.DataFrame(rows).sort_values("accuracy_mean", ascending=False)