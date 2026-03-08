import pandas as pd


def load_binary_data(csv_path: str) -> pd.DataFrame:
    df_raw = pd.read_csv(csv_path)

    print("shape:", df_raw.shape)
    print("columns:", df_raw.columns.tolist())

    if "String" in df_raw.columns:
        df_raw = df_raw.rename(columns={"String": "text"})
    elif "text" not in df_raw.columns:
        raise ValueError("❌ 'String' 또는 'text' 컬럼이 없습니다. 현재 컬럼: " + str(df_raw.columns.tolist()))

    if "label" not in df_raw.columns:
        raise ValueError("❌ 'label' 컬럼이 없습니다. 현재 컬럼: " + str(df_raw.columns.tolist()))

    df = df_raw[["text", "label"]].dropna().copy()
    df["label"] = df["label"].astype(int)

    print("\nlabel counts:\n", df["label"].value_counts(dropna=False))
    print("n_classes =", df["label"].nunique())
    print("unique labels =", sorted(df["label"].unique()))

    if df["label"].nunique() < 2:
        raise ValueError(
            "❌ label이 한 클래스(예: 전부 1) 뿐이라 5-fold CV 불가능합니다.\n"
            "=> NDP(=0) 포함된 파일로 바꾸세요."
        )

    print("\n✅ OK: binary labels detected. Proceed.")
    return df


def load_hierarchical_data(csv_path: str):
    df = pd.read_csv(csv_path)

    rename_map = {}
    for c in df.columns:
        c2 = c.strip()
        if c2.lower() == "string":
            rename_map[c] = "text"
        elif c2.lower() == "text":
            rename_map[c] = "text"
        elif c2.lower() == "predicate":
            rename_map[c] = "predicate"
        elif c2.lower() == "type":
            rename_map[c] = "type"
        elif c2 == "Type":
            rename_map[c] = "type"

    df = df.rename(columns=rename_map)

    needed = ["text", "predicate", "type"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. 현재 컬럼: {list(df.columns)}")

    df = df[needed].dropna().copy()
    df["text"] = df["text"].astype(str)
    df["predicate"] = df["predicate"].astype(str).str.strip()
    df["type"] = df["type"].astype(str).str.strip()

    predicates = sorted(df["predicate"].unique().tolist())
    types = sorted(df["type"].unique().tolist())

    pred2id = {p: i for i, p in enumerate(predicates)}
    id2pred = {i: p for p, i in pred2id.items()}

    type2id = {t: i for i, t in enumerate(types)}
    id2type = {i: t for t, i in type2id.items()}

    df["predicate_id"] = df["predicate"].map(pred2id).astype(int)
    df["type_id"] = df["type"].map(type2id).astype(int)

    g = df.groupby("predicate")["type"].nunique()
    bad = g[g > 1]
    if len(bad) > 0:
        raise ValueError(
            "Hierarchy violated: a predicate maps to multiple types. "
            f"문제 predicate: {bad.to_dict()}"
        )

    pred_to_type = df.groupby("predicate")["type"].first().to_dict()
    pred_id_to_type_id = {pred2id[p]: type2id[t] for p, t in pred_to_type.items()}

    meta = {
        "PREDICATES": predicates,
        "TYPES": types,
        "pred2id": pred2id,
        "id2pred": id2pred,
        "type2id": type2id,
        "id2type": id2type,
        "PRED_TO_TYPE": pred_to_type,
        "pred_id_to_type_id": pred_id_to_type_id,
    }

    print("shape:", df.shape)
    print("num_predicates:", len(predicates))
    print("num_types:", len(types))
    print("\n== Type distribution ==")
    print(df["type"].value_counts())
    print("\n== Predicate distribution ==")
    print(df["predicate"].value_counts())

    return df, meta