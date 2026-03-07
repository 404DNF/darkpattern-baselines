import pandas as pd


def load_and_validate_data(csv_path: str) -> pd.DataFrame:
    df_raw = pd.read_csv(csv_path)

    print("shape:", df_raw.shape)
    print("columns:", df_raw.columns.tolist())

    # text 컬럼 처리
    if "String" in df_raw.columns:
        df_raw = df_raw.rename(columns={"String": "text"})
    elif "text" not in df_raw.columns:
        raise ValueError("❌ 'String' 또는 'text' 컬럼이 없습니다. 현재 컬럼: " + str(df_raw.columns.tolist()))

    # label 컬럼 처리
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
            "=> NDP(=0) 포함된 파일로 바꾸세요. 지금 파일은 NDP가 빠져 있을 확률이 큽니다."
        )

    print("\n OK: binary labels detected. Proceed.")
    return df