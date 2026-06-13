import os
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")

REQUIRED_COLUMNS = ["TransactionID", "TransactionDT", "TransactionAmt", "card1", "isFraud"]


def load_and_merge() -> pd.DataFrame:
    transactions = pd.read_csv(os.path.join(RAW_DIR, "train_transaction.csv"))
    identity = pd.read_csv(os.path.join(RAW_DIR, "train_identity.csv"))
    df = transactions.merge(identity, on="TransactionID", how="left")
    print(f"Loaded {len(transactions):,} transactions, {len(identity):,} identity rows -> merged: {len(df):,} rows")
    return df


def validate(df: pd.DataFrame) -> bool:
    if df["TransactionID"].duplicated().any():
        raise ValueError("Duplicate TransactionIDs found")

    if "isFraud" not in df.columns:
        raise ValueError("Column 'isFraud' is missing")
    invalid_labels = set(df["isFraud"].unique()) - {0, 1}
    if invalid_labels:
        raise ValueError(f"isFraud contains unexpected values: {invalid_labels}")

    if not pd.api.types.is_numeric_dtype(df["TransactionDT"]):
        raise ValueError("TransactionDT is not numeric")
    if not (df["TransactionDT"].diff().dropna() >= 0).all():
        raise ValueError("TransactionDT is not monotonically increasing")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    fraud_rate = df["isFraud"].mean()
    print(f"Fraud rate: {fraud_rate:.4f} ({fraud_rate * 100:.2f}%)")
    return True


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    cutoff = int(len(df) * 0.8)
    train_df = df.iloc[:cutoff]
    test_df = df.iloc[cutoff:]

    print(f"Train class distribution:\n{train_df['isFraud'].value_counts(normalize=True).to_string()}")
    print(f"Test class distribution:\n{test_df['isFraud'].value_counts(normalize=True).to_string()}")
    return train_df, test_df


def save(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train_df.to_parquet(os.path.join(PROCESSED_DIR, "train.parquet"), index=False)
    test_df.to_parquet(os.path.join(PROCESSED_DIR, "test.parquet"), index=False)
    print(f"Saved train.parquet ({len(train_df):,} rows) and test.parquet ({len(test_df):,} rows)")


def main() -> None:
    df = load_and_merge()
    validate(df)
    train_df, test_df = split(df)
    save(train_df, test_df)

    total = len(df)
    fraud_rate = df["isFraud"].mean()
    print(f"\nSummary:")
    print(f"  Total rows  : {total:,}")
    print(f"  Fraud rate  : {fraud_rate:.4f}")
    print(f"  Train size  : {len(train_df):,}")
    print(f"  Test size   : {len(test_df):,}")


if __name__ == "__main__":
    main()
