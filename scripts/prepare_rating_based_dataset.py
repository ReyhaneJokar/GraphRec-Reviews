"""
Splits:
  rating >= positive_cutoff (4)          -> positive (train/val/test, leave-one/two-out per user)
  rating <= negative_cutoff (1)          -> negative (train only)
  negative_cutoff < rating < positive_cutoff (2,3) -> neutral (train only)

No ABSA/sentiment LLM step required
"""
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "reviewerID": "user_id",
        "asin": "item_id",
        "overall": "rating",
        "unixReviewTime": "timestamp",
        "reviewText": "text",
        "summary": "review_summary",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    required = ["user_id", "item_id", "rating", "timestamp"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns after normalization: {missing}. Available: {df.columns.tolist()}")
    return df


def clean_basic(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in ["user_id", "item_id", "rating", "timestamp", "text", "review_summary"] if c in df.columns]
    df = df[keep].copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["user_id", "item_id", "rating", "timestamp"]).copy()
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset=["user_id", "item_id"], keep="last").copy()
    return df.reset_index(drop=True)


def chronological_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Leave-last-one/two-out per user."""
    train_parts, val_parts, test_parts = [], [], []
    df = df.sort_values(["user_id", "timestamp", "row_index"])
    for _, g in df.groupby("user_id", sort=False):
        n = len(g)
        if n == 1:
            train_parts.append(g)
        elif n == 2:
            train_parts.append(g.iloc[:1])
            test_parts.append(g.iloc[1:])
        else:
            train_parts.append(g.iloc[:-2])
            val_parts.append(g.iloc[-2:-1])
            test_parts.append(g.iloc[-1:])

    def _concat(parts):
        return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0].copy()

    return _concat(train_parts), _concat(val_parts), _concat(test_parts)


def enforce_positive_train_coverage(df: pd.DataFrame, positive_cutoff: float, min_user_interactions: int, max_iter: int = 30) -> pd.DataFrame:
    """Every user/item kept must have >=1 positive edge in the train prefix,
    or data_loader.py's LabelEncoder.transform() on val/test hits an
    unseen-label error. Iterates to a fixed point."""
    df = df.copy()
    for it in range(1, max_iter + 1):
        before = len(df)
        df = df.groupby("user_id", group_keys=False).filter(lambda g: len(g) >= min_user_interactions)
        if df.empty:
            raise ValueError("All users removed during coverage filtering -- lower min_user_interactions or check rating thresholds.")

        probe = df.assign(row_index=np.arange(len(df)))  # temporary indices, only for stable sort
        train_probe, _, _ = chronological_split(probe)
        train_pos_probe = train_probe[train_probe["rating"] >= positive_cutoff]
        if train_pos_probe.empty:
            raise ValueError("No positive interactions remain in the train prefix.")

        keep_users = set(train_pos_probe["user_id"])
        keep_items = set(train_pos_probe["item_id"])
        new_df = df[df["user_id"].isin(keep_users) & df["item_id"].isin(keep_items)].copy()

        after = len(new_df)
        print(f"coverage iteration {it}: {before} -> {after}")
        df = new_df
        if after == before:
            break

    df = df.groupby("user_id", group_keys=False).filter(lambda g: len(g) >= min_user_interactions)
    return df.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Raw Amazon 5-core .json / .json.gz / .jsonl(.gz)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--min_user_interactions", type=int, default=3)
    ap.add_argument("--positive_cutoff", type=float, default=4.0)
    ap.add_argument("--negative_cutoff", type=float, default=1.0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading raw reviews...")
    df = pd.read_json(args.input, lines=True, compression="infer")
    df = normalize_columns(df)
    df = clean_basic(df)
    print("Rows after basic cleaning:", len(df))

    df["row_index"] = np.arange(len(df))  # provisional -- reassigned below after filtering
    df = enforce_positive_train_coverage(df, args.positive_cutoff, args.min_user_interactions)
    print("Rows after positive-coverage filtering:", len(df))

    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    df["row_index"] = np.arange(len(df))

    train_df, val_df, test_df = chronological_split(df)

    train_pos = train_df[train_df["rating"] >= args.positive_cutoff].copy()
    train_neg = train_df[train_df["rating"] <= args.negative_cutoff].copy()
    train_neu = train_df[(train_df["rating"] > args.negative_cutoff) & (train_df["rating"] < args.positive_cutoff)].copy()
    val_pos = val_df[val_df["rating"] >= args.positive_cutoff].copy()
    test_pos = test_df[test_df["rating"] >= args.positive_cutoff].copy()

    zero_pos_users = set(df["user_id"]) - set(train_pos["user_id"])
    zero_pos_items = set(df["item_id"]) - set(train_pos["item_id"])
    if zero_pos_users or zero_pos_items:
        raise AssertionError(
            f"Coverage filter did not converge: {len(zero_pos_users)} users / "
            f"{len(zero_pos_items)} items still have zero positive train edges."
        )

    cols = ["row_index", "user_id", "item_id", "rating", "timestamp", "text", "review_summary"]
    cols = [c for c in cols if c in df.columns]

    train_pos[cols].to_csv(outdir / "train_edges.csv", index=False, encoding="utf-8-sig")
    val_pos[cols].to_csv(outdir / "val_edges.csv", index=False, encoding="utf-8-sig")
    test_pos[cols].to_csv(outdir / "test_edges.csv", index=False, encoding="utf-8-sig")
    train_neg[cols].to_csv(outdir / "negative_edges.csv", index=False, encoding="utf-8-sig")
    train_neu[cols].to_csv(outdir / "neutral_edges.csv", index=False, encoding="utf-8-sig")
    df[cols].to_csv(outdir / "all_reviews_merged.csv", index=False, encoding="utf-8-sig")

    print("Done.")
    print(f"users={df['user_id'].nunique()} items={df['item_id'].nunique()} total_reviews={len(df)}")
    print(f"train_pos={len(train_pos)} val_pos={len(val_pos)} test_pos={len(test_pos)} "
          f"train_neg={len(train_neg)} train_neu={len(train_neu)}")
    for name in ["train_edges.csv", "val_edges.csv", "test_edges.csv", "negative_edges.csv",
                 "neutral_edges.csv", "all_reviews_merged.csv"]:
        print("Saved:", outdir / name)
    print("all_reviews_merged.csv is the canonical row_index source -- build embeddings from THIS file.")


if __name__ == "__main__":
    main()
