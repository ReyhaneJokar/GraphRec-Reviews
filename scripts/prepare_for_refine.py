import argparse
from pathlib import Path
import pandas as pd
from typing import Optional


def normalize_amazon_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}

    if "reviewerID" in df.columns:
        rename_map["reviewerID"] = "user_id"
    elif "user_id" in df.columns:
        rename_map["user_id"] = "user_id"

    if "asin" in df.columns:
        rename_map["asin"] = "item_id"
    elif "parent_asin" in df.columns:
        rename_map["parent_asin"] = "item_id"
    elif "item_id" in df.columns:
        rename_map["item_id"] = "item_id"

    if "overall" in df.columns:
        rename_map["overall"] = "rating"
    elif "rating" in df.columns:
        rename_map["rating"] = "rating"

    if "unixReviewTime" in df.columns:
        rename_map["unixReviewTime"] = "timestamp"
    elif "timestamp" in df.columns:
        rename_map["timestamp"] = "timestamp"

    if "reviewText" in df.columns:
        rename_map["reviewText"] = "text"
    elif "text" in df.columns:
        rename_map["text"] = "text"

    if "summary" in df.columns:
        rename_map["summary"] = "review_summary"

    if "verified" in df.columns:
        rename_map["verified"] = "verified_purchase"
    elif "verified_purchase" in df.columns:
        rename_map["verified_purchase"] = "verified_purchase"

    if "vote" in df.columns:
        rename_map["vote"] = "helpful_vote"
    elif "helpful_vote" in df.columns:
        rename_map["helpful_vote"] = "helpful_vote"

    if "image" in df.columns:
        rename_map["image"] = "images"
    elif "images" in df.columns:
        rename_map["images"] = "images"

    if "style" in df.columns:
        rename_map["style"] = "style"

    if "reviewTime" in df.columns:
        rename_map["reviewTime"] = "review_time"

    if "reviewerName" in df.columns:
        rename_map["reviewerName"] = "reviewer_name"

    df = df.rename(columns=rename_map)

    required = ["user_id", "item_id", "rating", "timestamp"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns after normalization: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    return df


def load_large_amazon_file(input_path: str, chunksize: int = 50000, max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Chunked reader for large Amazon JSON/JSON.GZ/JSONL files.
    Keeps only one chunk in memory at a time.
    """
    collected = []
    total_rows = 0

    reader = pd.read_json(
        input_path,
        lines=True,
        compression="infer",
        chunksize=chunksize,
    )

    for chunk_id, chunk in enumerate(reader):
        print(f"Processing chunk {chunk_id}")
        chunk = normalize_amazon_columns(chunk)

        keep_cols = [
            c for c in [
                "user_id",
                "item_id",
                "rating",
                "timestamp",
                "text",
                "review_summary",
                "verified_purchase",
                "helpful_vote",
                "images",
                "style",
                "review_time",
                "reviewer_name",
            ]
            if c in chunk.columns
        ]
        chunk = chunk[keep_cols].copy()

        chunk["rating"] = pd.to_numeric(chunk["rating"], errors="coerce")
        chunk["timestamp"] = pd.to_numeric(chunk["timestamp"], errors="coerce")
        chunk = chunk.dropna(subset=["user_id", "item_id", "rating", "timestamp"]).copy()

        collected.append(chunk)
        total_rows += len(chunk)
        print("total rows:", total_rows)

        if max_rows is not None and total_rows >= max_rows:
            break

    if len(collected) == 0:
        raise ValueError("No rows were loaded from the input file.")

    df = pd.concat(collected, ignore_index=True)
    return df


def clean_basic(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset=["user_id", "item_id"], keep="last").copy()
    return df


def k_core_filter(df: pd.DataFrame, user_col="user_id", item_col="item_id", k=5) -> pd.DataFrame:
    iteration = 0
    while True:
        iteration += 1
        before = len(df)

        user_count = df.groupby(user_col).size()
        item_count = df.groupby(item_col).size()

        valid_users = user_count[user_count >= k].index
        valid_items = item_count[item_count >= k].index

        df = df[df[user_col].isin(valid_users)]
        df = df[df[item_col].isin(valid_items)].copy()

        after = len(df)
        print(f"k-core iteration {iteration}: {before} -> {after}")

        if before == after:
            break

    return df


def ensure_positive_train_coverage(
    df: pd.DataFrame,
    positive_cutoff: float = 4.0,
    min_user_interactions: int = 3,
    max_iter: int = 30,
) -> pd.DataFrame:
    """
    Without changing the base code, ensure:
    - every user kept by preprocessing has at least one positive interaction
      in the train prefix (all but last 2 interactions per user)
    - every item kept by preprocessing also appears at least once in that
      positive train prefix

    This prevents unseen-label errors in the existing data_loader.
    """
    df = df.copy()

    for iteration in range(1, max_iter + 1):
        before = len(df)

        df = df.sort_values(["user_id", "timestamp"]).copy()

        # Keep only users with enough interactions for leave-one-out
        df = df.groupby("user_id", group_keys=False).filter(lambda g: len(g) >= min_user_interactions).copy()

        if len(df) == 0:
            raise ValueError("All users were removed before positive-coverage filtering.")

        # Build train prefix: all but last 2 interactions for each user
        prefix_parts = []
        for _, g in df.groupby("user_id"):
            if len(g) >= min_user_interactions:
                prefix_parts.append(g.iloc[:-2])

        if len(prefix_parts) == 0:
            raise ValueError("No train-prefix interactions remain after filtering.")

        train_prefix = pd.concat(prefix_parts, ignore_index=True)

        positive_prefix = train_prefix[train_prefix["rating"] >= positive_cutoff].copy()

        if len(positive_prefix) == 0:
            raise ValueError(
                "No positive interactions remain in the train prefix. "
                "Lower k-core or increase max_rows."
            )

        allowed_users = set(positive_prefix["user_id"].astype(str).unique())
        allowed_items = set(positive_prefix["item_id"].astype(str).unique())

        new_df = df[
            df["user_id"].astype(str).isin(allowed_users) &
            df["item_id"].astype(str).isin(allowed_items)
        ].copy()

        after = len(new_df)
        print(f"positive-coverage iteration {iteration}: {before} -> {after}")

        if after == before:
            df = new_df
            break

        df = new_df

    # Final sanity drop
    df = df.groupby("user_id", group_keys=False).filter(lambda g: len(g) >= min_user_interactions).copy()
    return df


def leave_one_out_split(df: pd.DataFrame):
    train_parts = []
    valid_parts = []
    test_parts = []

    df = df.sort_values(["user_id", "timestamp"]).copy()

    user_sizes = df.groupby("user_id").size()
    print("users >=3 interactions:", (user_sizes >= 3).sum())

    for _, g in df.groupby("user_id"):
        if len(g) < 3:
            continue
        train_parts.append(g.iloc[:-2])
        valid_parts.append(g.iloc[-2:-1])
        test_parts.append(g.iloc[-1:])

    if len(train_parts) == 0:
        raise ValueError("No users remain after split.")

    train_df = pd.concat(train_parts, ignore_index=True)
    valid_df = pd.concat(valid_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    return train_df, valid_df, test_df


def to_refine_format(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns={
        "user_id": "user_id:token",
        "item_id": "item_id:token",
        "rating": "rating:float",
        "timestamp": "timestamp:float",
    }).copy()

    return out[[
        "user_id:token",
        "item_id:token",
        "rating:float",
        "timestamp:float",
    ]]


def save_raw_reviews(df: pd.DataFrame, outdir: Path, dataset_name: str):
    raw_keep = [
        c for c in [
            "user_id",
            "item_id",
            "rating",
            "timestamp",
            "text",
            "review_summary",
            "verified_purchase",
            "helpful_vote",
            "images",
            "style",
            "review_time",
            "reviewer_name",
        ]
        if c in df.columns
    ]
    raw_df = df[raw_keep].copy()
    raw_df.to_csv(outdir / f"{dataset_name}_reviews_raw.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw Amazon .json.gz / .jsonl / .jsonl.gz file")
    parser.add_argument("--outdir", required=True, help="Output directory, e.g. E:/ReFINe/dataset/Video_Games")
    parser.add_argument("--dataset_name", required=True, help="Dataset name, e.g. Video_Games")
    parser.add_argument("--kcore", type=int, default=5, help="k-core threshold, default=5")
    parser.add_argument("--chunksize", type=int, default=50000, help="Chunk size for reading large files")
    parser.add_argument("--max_rows", type=int, default=None, help="Maximum raw rows to read (subset mode)")
    parser.add_argument("--positive_cutoff", type=float, default=4.0, help="Positive rating threshold")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    df = load_large_amazon_file(
        args.input,
        chunksize=args.chunksize,
        max_rows=args.max_rows
    )

    print("Rows loaded:", len(df))

    df = clean_basic(df)

    save_raw_reviews(df, outdir, args.dataset_name)
    print("Saved raw reviews")

    df = k_core_filter(df, k=args.kcore)
    print("After k-core:", len(df))

    df = ensure_positive_train_coverage(
        df,
        positive_cutoff=args.positive_cutoff,
        min_user_interactions=3
    )
    print("After positive-train coverage filter:", len(df))

    train_df, valid_df, test_df = leave_one_out_split(df)

    train_out = to_refine_format(train_df)
    valid_out = to_refine_format(valid_df)
    test_out = to_refine_format(test_df)

    train_out.to_csv(outdir / f"{args.dataset_name}_train_original.csv", index=False)
    train_out.to_csv(outdir / f"{args.dataset_name}_train_augment.csv", index=False)
    valid_out.to_csv(outdir / f"{args.dataset_name}_validation.csv", index=False)
    test_out.to_csv(outdir / f"{args.dataset_name}_test.csv", index=False)

    print("\nFinished.")
    print("Train:", len(train_out))
    print("Valid:", len(valid_out))
    print("Test :", len(test_out))
    print("Output:", outdir)


if __name__ == "__main__":
    main()