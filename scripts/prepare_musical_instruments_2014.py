import argparse
from pathlib import Path
import pandas as pd


def load_raw_reviews(input_path: str) -> pd.DataFrame:
    return pd.read_json(input_path, lines=True, compression="infer")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}

    if "reviewerID" in df.columns:
        rename_map["reviewerID"] = "user_id"
    if "asin" in df.columns:
        rename_map["asin"] = "item_id"
    if "overall" in df.columns:
        rename_map["overall"] = "rating"
    if "unixReviewTime" in df.columns:
        rename_map["unixReviewTime"] = "timestamp"
    if "reviewText" in df.columns:
        rename_map["reviewText"] = "text"
    if "summary" in df.columns:
        rename_map["summary"] = "review_summary"

    df = df.rename(columns=rename_map)

    required = ["user_id", "item_id", "rating", "timestamp"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Available: {df.columns.tolist()}")

    return df


def clean_basic(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [c for c in ["user_id", "item_id", "rating", "timestamp", "text", "review_summary"] if c in df.columns]
    df = df[keep_cols].copy()

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["user_id", "item_id", "rating", "timestamp"]).copy()

    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset=["user_id", "item_id"], keep="last").copy()
    return df


def leave_one_out_split(df: pd.DataFrame):
    train_parts, valid_parts, test_parts = [], [], []
    df = df.sort_values(["user_id", "timestamp"]).copy()

    for _, g in df.groupby("user_id"):
        if len(g) < 3:
            continue
        train_parts.append(g.iloc[:-2])
        valid_parts.append(g.iloc[-2:-1])
        test_parts.append(g.iloc[-1:])

    if not train_parts:
        raise ValueError("No users left after split.")

    train_df = pd.concat(train_parts, ignore_index=True)
    valid_df = pd.concat(valid_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    return train_df, valid_df, test_df


def closure_filter_until_stable(df: pd.DataFrame, pos_threshold: float = 4.0, max_iter: int = 20):
    """
    Iteratively keep only users/items that survive in the positive train prefix.
    This makes the CSVs compatible with the current ReFINe_plus data_loader.
    """
    for it in range(max_iter):
        before = len(df)

        train_df, valid_df, test_df = leave_one_out_split(df)
        train_pos = train_df[train_df["rating"] >= pos_threshold].copy()

        if len(train_pos) == 0:
            raise ValueError(
                "No positive interactions remain in train. "
                "Try a different subset or relax filtering."
            )

        keep_users = set(train_pos["user_id"].astype(str).unique())
        keep_items = set(train_pos["item_id"].astype(str).unique())

        new_df = df[
            df["user_id"].astype(str).isin(keep_users) &
            df["item_id"].astype(str).isin(keep_items)
        ].copy()

        after = len(new_df)
        print(f"closure iteration {it+1}: {before} -> {after}")

        if after == before:
            return new_df

        df = new_df

    return df


def to_refine_format(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns={
        "user_id": "user_id:token",
        "item_id": "item_id:token",
        "rating": "rating:float",
        "timestamp": "timestamp:float",
    }).copy()
    return out[["user_id:token", "item_id:token", "rating:float", "timestamp:float"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--dataset_name", required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_raw_reviews(args.input)
    df = normalize_columns(df)
    df = clean_basic(df)

    # Save raw reviews for future ABSA / LLM refinement
    raw_keep = [c for c in ["user_id", "item_id", "rating", "timestamp", "text", "review_summary"] if c in df.columns]
    df[raw_keep].to_csv(outdir / f"{args.dataset_name}_reviews_raw.csv", index=False)

    # Strict filtering so the base code can fit encoders on positive train safely
    df = closure_filter_until_stable(df, pos_threshold=4.0, max_iter=20)
    print("After closure filtering:", len(df))

    train_df, valid_df, test_df = leave_one_out_split(df)

    # Optional final sanity: remove any rows whose user/item does not exist in positive train
    train_pos = train_df[train_df["rating"] >= 4.0].copy()
    keep_users = set(train_pos["user_id"].astype(str).unique())
    keep_items = set(train_pos["item_id"].astype(str).unique())

    train_df = train_df[
        train_df["user_id"].astype(str).isin(keep_users) &
        train_df["item_id"].astype(str).isin(keep_items)
    ].copy()
    valid_df = valid_df[
        valid_df["user_id"].astype(str).isin(keep_users) &
        valid_df["item_id"].astype(str).isin(keep_items)
    ].copy()
    test_df = test_df[
        test_df["user_id"].astype(str).isin(keep_users) &
        test_df["item_id"].astype(str).isin(keep_items)
    ].copy()

    train_out = to_refine_format(train_df)
    valid_out = to_refine_format(valid_df)
    test_out = to_refine_format(test_df)

    train_out.to_csv(outdir / f"{args.dataset_name}_train_original.csv", index=False)
    train_out.to_csv(outdir / f"{args.dataset_name}_train_augment.csv", index=False)
    valid_out.to_csv(outdir / f"{args.dataset_name}_validation.csv", index=False)
    test_out.to_csv(outdir / f"{args.dataset_name}_test.csv", index=False)

    print("Done.")
    print("Train:", len(train_out))
    print("Valid:", len(valid_out))
    print("Test :", len(test_out))
    print("Saved to:", outdir)

    # sanity check for unseen labels
    train_users = set(train_pos["user_id"].astype(str))
    train_items = set(train_pos["item_id"].astype(str))
    for name, check_df in [("valid", valid_out), ("test", test_out)]:
        bad_u = (~check_df["user_id:token"].astype(str).isin(train_users)).sum()
        bad_i = (~check_df["item_id:token"].astype(str).isin(train_items)).sum()
        print(name, "unseen users:", int(bad_u), "| unseen items:", int(bad_i))


if __name__ == "__main__":
    main()