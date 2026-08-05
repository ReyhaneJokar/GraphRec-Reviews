#!/usr/bin/env python
r"""
Build a parallel graph_ready folder for an A/B ablation test:
  train_edges.csv = POSITIVE UNION NEGATIVE (mirrors the literal old
  base-repo data_loader.py, which builds its structural graph from
  df_train[(rating>=4)|(rating<=1)]).

Everything else (val_edges.csv, test_edges.csv, negative_edges.csv,
neutral_edges.csv, edge_features.npy, review_embeddings*.npy,
aspect_vector_matrix.npy, user_map.json, item_map.json) is copied
UNCHANGED from the source graph_ready/ folder, so the coverage-filtered
dataset, split, and vocabulary are held exactly constant. This isolates
ONE variable: whether negative edges participate in LightGCN message
passing.

No changes to main.py / data_loader.py are needed: main.py already
filters pos_neg_edge_rate>=4 for the BPR loss regardless of what
train_edges.csv contains, and data_loader.py loads train_edges.csv
verbatim as the structural edge_index.

Usage:
    python build_posneg_graph_ablation.py \
        --source_dir "E:\ReFINe\dataset\Musical_Instruments_2014\graph_ready" \
        --output_dir "E:\ReFINe\dataset\Musical_Instruments_2014\graph_ready_posneg_ablation"
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd


COPY_AS_IS = [
    "val_edges.csv",
    "test_edges.csv",
    "negative_edges.csv",
    "neutral_edges.csv",
    "user_map.json",
    "item_map.json",
    "edge_features.npy",
    "review_embeddings.npy",
    "review_embeddings_text_only.npy",
    "aspect_vector_matrix.npy",
    "edge_feature_manifest.json",
    "aspect_vocab.json",
    "aspect_vocab_filtered.json",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", required=True, help="Existing graph_ready/ folder")
    parser.add_argument("--output_dir", required=True, help="New ablation folder to create")
    args = parser.parse_args()

    src = Path(args.source_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_pos_path = src / "train_edges.csv"
    train_neg_path = src / "negative_edges.csv"

    if not train_pos_path.exists():
        raise FileNotFoundError(train_pos_path)
    if not train_neg_path.exists():
        raise FileNotFoundError(train_neg_path)

    train_pos = pd.read_csv(train_pos_path)
    train_neg = pd.read_csv(train_neg_path)

    assert set(train_pos["row_index"]).isdisjoint(set(train_neg["row_index"])), \
        "train_edges.csv and negative_edges.csv overlap -- unexpected, stop and investigate"

    train_pos_neg = pd.concat([train_pos, train_neg], ignore_index=True)
    train_pos_neg.to_csv(out / "train_edges.csv", index=False, encoding="utf-8-sig")

    for fname in COPY_AS_IS:
        src_file = src / fname
        if src_file.exists():
            shutil.copy2(src_file, out / fname)

    print("Done.")
    print(f"train_edges.csv (positive only, source): {len(train_pos)} rows")
    print(f"negative_edges.csv (unchanged, also copied): {len(train_neg)} rows")
    print(f"train_edges.csv (positive UNION negative, ablation): {len(train_pos_neg)} rows")
    print(f"Ablation folder ready at: {out}")
    print("Run main.py with --project_dir pointing at this folder.")


if __name__ == "__main__":
    main()
