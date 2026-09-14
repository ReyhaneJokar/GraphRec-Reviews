import argparse
import json
import re
from pathlib import Path

import pandas as pd

RUN_NAME_RE = re.compile(r"^(?P<track>.+)_seed(?P<seed>\d+)$")

METRIC_KEYS = [f"{m}@{k}" for k in (5, 10, 15, 20) for m in ("Precision", "Recall", "NDCG")]


def parse_run_name(run_name: str):
    m = RUN_NAME_RE.match(run_name)
    if m:
        return m.group("track"), int(m.group("seed"))
    return run_name, None


def load_metrics_file(path: Path, dataset: str, split: str) -> dict:
    with path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    suffix = f"_{split}_metrics.json" if split == "test" else "_best_val_metrics.json"
    run_name = path.name[: -len(suffix)] if path.name.endswith(suffix) else path.stem
    track, seed = parse_run_name(run_name)

    row = {
        "dataset": dataset,
        "run_name": run_name,
        "track": track,
        "seed": seed,
        "split": split,
        "source_file": str(path),
    }
    for key in METRIC_KEYS:
        row[key] = metrics.get(key, None)

    for key, value in metrics.items():
        if key not in METRIC_KEYS:
            row[key] = value

    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True, help="Root folder containing one subfolder per dataset")
    ap.add_argument("--output", default="all_results.csv")
    args = ap.parse_args()

    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        raise FileNotFoundError(f"{result_dir} does not exist")

    rows = []
    test_files = sorted(result_dir.glob("*/*_test_metrics.json"))
    val_files = sorted(result_dir.glob("*/*_best_val_metrics.json"))

    for path in test_files:
        dataset = path.parent.name
        rows.append(load_metrics_file(path, dataset, split="test"))

    for path in val_files:
        dataset = path.parent.name
        rows.append(load_metrics_file(path, dataset, split="val"))

    if not rows:
        raise FileNotFoundError(
            f"No *_test_metrics.json or *_best_val_metrics.json files found under {result_dir}. "
            f"Check --result_dir points at the folder that CONTAINS the per-dataset subfolders."
        )

    df = pd.DataFrame(rows)
    front_cols = ["dataset", "track", "seed", "split", "run_name"] + METRIC_KEYS
    other_cols = [c for c in df.columns if c not in front_cols + ["source_file"]]
    df = df[front_cols + other_cols + ["source_file"]]
    df = df.sort_values(["dataset", "split", "track", "seed"], na_position="first").reset_index(drop=True)

    output_path = Path(args.output)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Collected {len(df)} rows from {len(test_files)} test files + {len(val_files)} val files.")
    print(f"Saved: {output_path}")
    print()
    print("Per dataset / track / split run counts (sanity check -- should match how many seeds you actually ran):")
    print(df.groupby(["dataset", "split", "track"]).size().to_string())

    print()
    print("Quick mean preview (test split, Recall@20 / NDCG@20, across whatever seeds are present):")
    test_df = df[df["split"] == "test"]
    if not test_df.empty:
        preview = test_df.groupby(["dataset", "track"])[["Recall@20", "NDCG@20"]].agg(["mean", "std", "count"])
        print(preview.to_string())


if __name__ == "__main__":
    main()
