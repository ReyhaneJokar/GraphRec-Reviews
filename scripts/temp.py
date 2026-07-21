import pandas as pd
import numpy as np
from pathlib import Path

base = Path(r"E:\ReFINe\dataset\Musical_Instruments_2014\graph_ready")

USER_CANDIDATES = ["user_id:token", "user_id", "user", "uid"]
ITEM_CANDIDATES = ["item_id:token", "item_id", "item", "iid"]
RATING_CANDIDATES = ["rating:float", "rating", "score"]

def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

for name in [
    "train_edges.csv",
    "val_edges.csv",
    "test_edges.csv",
    "negative_edges.csv",
    "neutral_edges.csv",
]:
    df = pd.read_csv(base / name)
    ucol = find_col(df, USER_CANDIDATES)
    icol = find_col(df, ITEM_CANDIDATES)
    rcol = find_col(df, RATING_CANDIDATES)

    print(name)
    print("  rows:", len(df))
    print("  columns:", list(df.columns))
    print("  user_col:", ucol)
    print("  item_col:", icol)
    print("  rating_col:", rcol)

    if ucol is not None:
        print("  unique users:", df[ucol].nunique())
    if icol is not None:
        print("  unique items:", df[icol].nunique())
    print()

feat = np.load(base / "edge_features.npy")
print("edge_features.npy shape:", feat.shape)
print("edge_features.npy dtype:", feat.dtype)