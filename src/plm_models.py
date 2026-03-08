import os
import random
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from src.config import SEED, N_SPLITS, MAX_LENGTH
from src.metrics import (
    compute_metrics_binary,
    compute_metrics_multiclass,
    per_class_metrics,
    summarize_fold_metrics,
    softmax_np,
)

# ----------------
# Device
# ----------------
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


# ----------------
# Seed
# ----------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


seed_everything(SEED)
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)


# ----------------
# PLM specs
# ----------------

PLM_SPECS = {
    "BERT-base": {"ckpt": "bert-base-uncased", "bs": 16, "lr": 4e-5, "dropout": 0.1, "epochs": 5},
    "BERT-large": {"ckpt": "bert-large-uncased", "bs": 32, "lr": 3e-5, "dropout": 0.1, "epochs": 5},
    "RoBERTa-base": {"ckpt": "roberta-base", "bs": 32, "lr": 3e-5, "dropout": 0.1, "epochs": 5},
    "RoBERTa-large": {"ckpt": "roberta-large", "bs": 16, "lr": 3e-5, "dropout": 0.1, "epochs": 5},
    "ALBERT-base": {"ckpt": "albert-base-v2", "bs": 16, "lr": 3e-5, "dropout": 0.1, "epochs": 5},
    "ALBERT-large": {"ckpt": "albert-large-v2", "bs": 16, "lr": 5e-5, "dropout": 0.1, "epochs": 5},
    "XLNet-base": {"ckpt": "xlnet-base-cased", "bs": 8, "lr": 2e-5, "dropout": 0.1, "epochs": 5},
    "XLNet-large": {"ckpt": "xlnet-large-cased", "bs": 4, "lr": 4e-5, "dropout": 0.1, "epochs": 5},
}


# ----------------
# Safe loading utils
# ----------------
def set_dropout_in_config(cfg, dropout):
    for attr in [
        "hidden_dropout_prob",
        "attention_probs_dropout_prob",
        "classifier_dropout",
        "dropout",
    ]:
        if hasattr(cfg, attr) and getattr(cfg, attr) is not None:
            setattr(cfg, attr, dropout)
    return cfg


def _load_tokenizer_safe(ckpt, model_key, local_files_only=False):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            ckpt,
            use_fast=True,
            local_files_only=local_files_only,
        )
        return tokenizer
    except Exception as e:
        print(f"⚠️ {model_key} skipped: cannot load tokenizer. ({type(e).__name__}) {e}")
        return None


def _load_config_safe(ckpt, model_key, num_labels, local_files_only=False):
    try:
        cfg = AutoConfig.from_pretrained(
            ckpt,
            num_labels=num_labels,
            local_files_only=local_files_only,
        )
        return cfg
    except Exception as e:
        print(f"⚠️ {model_key} skipped: cannot load config. ({type(e).__name__}) {e}")
        return None


def _load_model_safe(ckpt, cfg, model_key, local_files_only=False):
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            ckpt,
            config=cfg,
            use_safetensors=True,
            local_files_only=local_files_only,
        ).to(device)
        return model
    except Exception as e:
        print(f"⚠️ {model_key} skipped: cannot load with safetensors. ({type(e).__name__}) {e}")
        return None


# =========================================================
# 1) Binary PLM
# =========================================================
def trainer_compute_metrics_binary(eval_pred):
    logits, labels = eval_pred
    probs = softmax_np(logits)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return compute_metrics_binary(labels, preds, probs)


def run_binary_plm_5fold(df, model_key, out_root, local_files_only=False):
    seed_everything(SEED)

    X = df["text"].tolist()
    y = df["label"].astype(int).tolist()

    spec = PLM_SPECS[model_key]
    ckpt, bs, lr, dr, epochs = (
        spec["ckpt"],
        spec["bs"],
        spec["lr"],
        spec["dropout"],
        spec["epochs"],
    )

    tokenizer = _load_tokenizer_safe(
        ckpt,
        model_key,
        local_files_only=local_files_only,
    )
    if tokenizer is None:
        return None

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    fold_metrics = []

    for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        print(f"   - Fold {fold}/{N_SPLITS}")

        train_df = pd.DataFrame({
            "text": [X[i] for i in tr],
            "label": [y[i] for i in tr],
        })
        test_df = pd.DataFrame({
            "text": [X[i] for i in te],
            "label": [y[i] for i in te],
        })

        train_ds = Dataset.from_pandas(train_df, preserve_index=False)
        eval_ds = Dataset.from_pandas(test_df, preserve_index=False)

        def tok(batch):
            return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

        train_ds = train_ds.map(tok, batched=True, remove_columns=["text"])
        eval_ds = eval_ds.map(tok, batched=True, remove_columns=["text"])

        cfg = _load_config_safe(
            ckpt,
            model_key,
            num_labels=2,
            local_files_only=local_files_only,
        )
        if cfg is None:
            return None

        cfg = set_dropout_in_config(cfg, dr)

        model = _load_model_safe(
            ckpt,
            cfg,
            model_key,
            local_files_only=local_files_only,
        )
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
            seed=SEED,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=data_collator,
            compute_metrics=trainer_compute_metrics_binary,
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
        elif device == "mps":
            torch.mps.empty_cache()

    return summarize_fold_metrics(fold_metrics)


def run_binary_plm_models(df, model_keys, save_dir, local_files_only=False):
    os.makedirs(save_dir, exist_ok=True)

    rows = []
    partial_path = os.path.join(save_dir, "plm_results_partial.csv")

    for k in model_keys:
        print(f"\n🚀 Running Binary | {k}")
        summ = run_binary_plm_5fold(
            df=df,
            model_key=k,
            out_root=save_dir,
            local_files_only=local_files_only,
        )

        if summ is None:
            print(f"⏭️ skipped: {k}")
            continue

        row = {"model": k, **summ}
        rows.append(row)

        pd.DataFrame(rows).to_csv(partial_path, index=False)
        print("💾 saved partial:", partial_path)

    if len(rows) == 0:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("accuracy_mean", ascending=False)


# =========================================================
# 2) Predicate / 3) Type Hierarchical PLM
# =========================================================
def build_hf_dataset(df_part, tokenizer, label_col: str):
    ds = Dataset.from_pandas(
        df_part[["text", label_col]].rename(columns={label_col: "labels"}),
        preserve_index=False,
    )

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    ds = ds.map(tok, batched=True, remove_columns=["text"])
    return ds


def run_plm_5fold_multiclass(
    df,
    task,
    model_key,
    out_root,
    meta,
    save_classwise=True,
    local_files_only=False,
):
    seed_everything(SEED)

    spec = PLM_SPECS[model_key]
    ckpt, bs, lr, dr, epochs = (
        spec["ckpt"],
        spec["bs"],
        spec["lr"],
        spec["dropout"],
        spec["epochs"],
    )

    if task == "predicate":
        label_col = "predicate_id"
        num_labels = len(meta["PREDICATES"])
        id2name = meta["id2pred"]
    elif task == "type":
        label_col = "type_id"
        num_labels = len(meta["TYPES"])
        id2name = meta["id2type"]
    else:
        raise ValueError("task must be 'predicate' or 'type'")

    tokenizer = _load_tokenizer_safe(
        ckpt,
        model_key,
        local_files_only=local_files_only,
    )
    if tokenizer is None:
        return None

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    y = df[label_col].astype(int).values

    fold_metrics = []
    classwise_rows = []

    for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        print(f"   - Fold {fold}/{N_SPLITS}")

        train_df = df.iloc[tr].reset_index(drop=True)
        test_df = df.iloc[te].reset_index(drop=True)

        train_ds = build_hf_dataset(train_df, tokenizer, label_col)
        eval_ds = build_hf_dataset(test_df, tokenizer, label_col)

        cfg = _load_config_safe(
            ckpt,
            model_key,
            num_labels=num_labels,
            local_files_only=local_files_only,
        )
        if cfg is None:
            return None

        cfg = set_dropout_in_config(cfg, dr)

        model = _load_model_safe(
            ckpt,
            cfg,
            model_key,
            local_files_only=local_files_only,
        )
        if model is None:
            return None

        args = TrainingArguments(
            output_dir=f"{out_root}/{task}/{model_key.replace('/', '_')}/fold{fold}",
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
            seed=SEED,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=data_collator,
        )

        trainer.train()

        pred_out = trainer.predict(eval_ds)
        logits = pred_out.predictions
        y_true = pred_out.label_ids
        y_pred = np.argmax(logits, axis=-1)

        m = compute_metrics_multiclass(y_true, y_pred)

        if task == "predicate":
            true_type = test_df["type_id"].astype(int).values
            pred_type = np.array(
                [meta["pred_id_to_type_id"][int(pid)] for pid in y_pred],
                dtype=int,
            )
            m["derived_type_accuracy"] = float(accuracy_score(true_type, pred_type))
            m["derived_type_macro_f1"] = float(
                f1_score(true_type, pred_type, average="macro", zero_division=0)
            )

        fold_metrics.append(m)

        if save_classwise:
            cw = per_class_metrics(y_true, y_pred, id2name)
            cw["task"] = task
            cw["model"] = model_key
            cw["fold"] = fold
            classwise_rows.append(cw)

        del model, trainer

        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

    summ = summarize_fold_metrics(fold_metrics)

    classwise_df = None
    if save_classwise and len(classwise_rows) > 0:
        classwise_df = pd.concat(classwise_rows, ignore_index=True)

    return summ, classwise_df


def load_done_set(partial_csv):
    if not os.path.exists(partial_csv):
        return set(), pd.DataFrame()

    prev = pd.read_csv(partial_csv)
    if len(prev) == 0:
        return set(), prev

    done = set(zip(prev["task"].astype(str), prev["model"].astype(str)))
    return done, prev


def run_hierarchical_plm_models(
    df,
    meta,
    run_tasks,
    model_keys,
    save_dir,
    save_classwise=True,
    local_files_only=False,
):
    os.makedirs(save_dir, exist_ok=True)

    partial_csv = os.path.join(save_dir, "hier_plm_results_partial.csv")
    final_csv = os.path.join(save_dir, "hier_plm_results_final.csv")
    classwise_csv = os.path.join(save_dir, "hier_plm_classwise_long.csv")

    done_set, prev_df = load_done_set(partial_csv)
    print("done combos:", len(done_set))

    rows = prev_df.to_dict("records") if len(prev_df) > 0 else []

    classwise_all = []
    if save_classwise and os.path.exists(classwise_csv):
        try:
            classwise_all.append(pd.read_csv(classwise_csv))
            print("loaded existing classwise:", classwise_csv)
        except Exception as e:
            print("classwise load failed (ignore):", e)

    for task in run_tasks:
        for model_key in model_keys:
            combo = (task, model_key)
            if combo in done_set:
                print(f"⏭️ skip done: task={task}, model={model_key}")
                continue

            print(f"\n🚀 Running task={task} | model={model_key}")
            out = run_plm_5fold_multiclass(
                df=df,
                task=task,
                model_key=model_key,
                out_root=save_dir,
                meta=meta,
                save_classwise=save_classwise,
                local_files_only=local_files_only,
            )

            if out is None:
                print(f"⏭️ skipped: task={task}, model={model_key}")
                continue

            summ, classwise_df = out
            row = {"task": task, "model": model_key, **summ}
            rows.append(row)

            pd.DataFrame(rows).to_csv(partial_csv, index=False)
            print("💾 saved partial:", partial_csv)

            if save_classwise and classwise_df is not None:
                classwise_all.append(classwise_df)
                pd.concat(classwise_all, ignore_index=True).to_csv(classwise_csv, index=False)
                print("💾 saved classwise:", classwise_csv)

    res = pd.DataFrame(rows)

    if len(res) > 0:
        sort_cols = ["task", "macro_f1_mean"] if "macro_f1_mean" in res.columns else ["task"]
        ascending = [True, False] if "macro_f1_mean" in res.columns else [True]
        res = res.sort_values(sort_cols, ascending=ascending)
        res.to_csv(final_csv, index=False)
        print("✅ saved final:", final_csv)
    else:
        print("No results to save.")

    return res


def pretty_view_hierarchical(df_res):
    show_cols = [c for c in df_res.columns if c in [
        "task", "model",
        "accuracy_mean", "accuracy_std",
        "macro_f1_mean", "macro_f1_std",
        "weighted_f1_mean", "weighted_f1_std",
        "derived_type_accuracy_mean", "derived_type_accuracy_std",
        "derived_type_macro_f1_mean", "derived_type_macro_f1_std",
    ]]
    return df_res[show_cols]