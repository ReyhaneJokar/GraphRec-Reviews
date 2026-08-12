#!/usr/bin/env python

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import data_loader
from model import ReFINe_plus


def infer_checkpoint_edge_dim(state_dict):
    """
    Infer edge_attr input dimension directly from checkpoint.

    Current architecture:
        edge_attr_proj.0.weight -> [edge_attr_dim]

    Older architecture may have:
        edge_attr_proj.0.weight -> [hidden, edge_attr_dim]
    """
    key = "edge_attr_proj.0.weight"

    if key in state_dict:
        shape = tuple(state_dict[key].shape)

        if len(shape) == 1:
            return int(shape[0])

        if len(shape) == 2:
            return int(shape[1])

    # Fallback
    for key in (
        "edge_attr_proj.1.weight",
        "edge_attr_proj.0.weight",
    ):
        if key in state_dict:
            shape = tuple(state_dict[key].shape)
            if len(shape) == 2:
                return int(shape[1])

    return 0


def infer_checkpoint_embedding_dim(state_dict):
    key = "embedding.weight"

    if key not in state_dict:
        raise RuntimeError(
            "embedding.weight not found in checkpoint."
        )

    shape = tuple(state_dict[key].shape)

    if len(shape) != 2:
        raise RuntimeError(
            f"Unexpected embedding.weight shape: {shape}"
        )

    return int(shape[1])


def infer_checkpoint_num_layers(state_dict):
    layer_ids = set()

    for key in state_dict.keys():
        if key.startswith("convs."):
            parts = key.split(".")
            if len(parts) >= 2 and parts[1].isdigit():
                layer_ids.add(int(parts[1]))

    if not layer_ids:
        raise RuntimeError(
            "Could not infer GNN layer count from checkpoint."
        )

    return max(layer_ids) + 1


def load_text_only_edge_embeddings(project_dir: Path, embedding_file: Path, train_df: pd.DataFrame, negative_df: pd.DataFrame,):

    if not embedding_file.is_absolute():
        embedding_file = project_dir / embedding_file

    if not embedding_file.exists():
        raise FileNotFoundError(
            f"Text-only embedding file not found:\n{embedding_file}"
        )

    embeddings = np.load(embedding_file)

    if embeddings.ndim != 2:
        raise ValueError(
            f"Embedding array must be 2D, got {embeddings.shape}"
        )

    if "row_index" not in train_df.columns:
        raise ValueError(
            "train_edges.csv must contain row_index."
        )

    if "row_index" not in negative_df.columns:
        raise ValueError(
            "negative_edges.csv must contain row_index."
        )

    pos_rows = train_df["row_index"].astype(int).to_numpy()
    neg_rows = negative_df["row_index"].astype(int).to_numpy()

    max_needed = max(
        int(pos_rows.max()) if len(pos_rows) else -1,
        int(neg_rows.max()) if len(neg_rows) else -1,
    )

    if embeddings.shape[0] <= max_needed:
        raise ValueError(
            f"Embedding file has {embeddings.shape[0]} rows, "
            f"but max required row_index is {max_needed}."
        )

    pos_embeddings = embeddings[pos_rows].astype(
        np.float32,
        copy=False,
    )

    neg_embeddings = embeddings[neg_rows].astype(
        np.float32,
        copy=False,
    )

    return embeddings, pos_embeddings, neg_embeddings, embedding_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", required=True,)
    ap.add_argument("--checkpoint", required=True,)
    ap.add_argument("--base_embeddings", required=True,)
    ap.add_argument("--edge_attr_file", required=True, help="Explicit text-only review embedding file. Example: review_embeddings_text_only.npy",)
    ap.add_argument("--llm_labels_csv", required=True, help="Remapped LLM negative-confidence labels CSV.",)
    ap.add_argument("--embedding_dim", type=int, default=None,)
    ap.add_argument("--layers", type=int, default=None,)
    ap.add_argument("--output", required=True,)
    args = ap.parse_args()

    project_dir = Path(args.project_dir)
    checkpoint_path = Path(args.checkpoint)
    base_embeddings_path = Path(args.base_embeddings)
    llm_labels_path = Path(args.llm_labels_csv)
    output_path = Path(args.output)

    for p, name in [
        (checkpoint_path, "checkpoint"),
        (base_embeddings_path, "base embeddings"),
        (llm_labels_path, "LLM labels"),
        (project_dir / "train_edges.csv", "train_edges.csv"),
        (project_dir / "negative_edges.csv", "negative_edges.csv"),
    ]:
        if not p.exists():
            raise FileNotFoundError(
                f"{name} not found:\n{p}"
            )

    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    checkpoint_edge_dim = infer_checkpoint_edge_dim(state_dict)
    checkpoint_embedding_dim = infer_checkpoint_embedding_dim(state_dict)
    checkpoint_layers = infer_checkpoint_num_layers(state_dict)

    embedding_dim = (
        checkpoint_embedding_dim
        if args.embedding_dim is None
        else args.embedding_dim
    )

    layers = (
        checkpoint_layers
        if args.layers is None
        else args.layers
    )

    print("=" * 80)
    print("Build distillation dataset")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Checkpoint embedding_dim: {checkpoint_embedding_dim}")
    print(f"Checkpoint layers: {checkpoint_layers}")
    print(f"Checkpoint edge_attr_dim: {checkpoint_edge_dim}")

    if embedding_dim != checkpoint_embedding_dim:
        raise RuntimeError(
            f"embedding_dim mismatch: "
            f"checkpoint={checkpoint_embedding_dim}, "
            f"requested={embedding_dim}"
        )

    if layers != checkpoint_layers:
        raise RuntimeError(
            f"layers mismatch: "
            f"checkpoint={checkpoint_layers}, "
            f"requested={layers}"
        )

    neg_df = pd.read_csv(project_dir / "negative_edges.csv")
    pos_df = pd.read_csv(project_dir / "train_edges.csv")

    if "row_index" not in neg_df.columns:
        raise ValueError("negative_edges.csv must contain row_index.")

    if "row_index" not in pos_df.columns:
        raise ValueError("train_edges.csv must contain row_index.")

    neg_df["row_index"] = neg_df["row_index"].astype(int)
    pos_df["row_index"] = pos_df["row_index"].astype(int)

    llm_df = pd.read_csv(llm_labels_path)

    required_llm_cols = {"row_index", "negative_confidence"}
    missing = required_llm_cols - set(llm_df.columns)

    if missing:
        raise ValueError(
            f"LLM labels file missing columns: {sorted(missing)}"
        )

    llm_df = llm_df[
        ["row_index", "negative_confidence"]
    ].copy()

    llm_df["row_index"] = llm_df["row_index"].astype(int)

    duplicate_count = llm_df["row_index"].duplicated().sum()

    if duplicate_count:
        raise ValueError(
            f"LLM labels contain {duplicate_count} duplicated row_index values. "
            "Each negative review must have at most one teacher label."
        )

    llm_df["negative_confidence"] = pd.to_numeric(
        llm_df["negative_confidence"],
        errors="coerce",
    )

    llm_df = llm_df.dropna(
        subset=["negative_confidence"]
    )

    llm_df["negative_confidence"] = llm_df[
        "negative_confidence"
    ].clip(0.0, 1.0)

    data, data_neg, _ = data_loader.data_loading(
        str(project_dir),
        load_val_or_test="val",
    )

    num_users = int(data["user"].num_nodes)
    num_items = int(data["item"].num_nodes)

    print(f"Users: {num_users}")
    print(f"Items: {num_items}")
    print(f"Positive train edges: {len(pos_df)}")
    print(f"Negative train edges: {len(neg_df)}")

    pos_edge_index = data[
        "user", "rates", "item"
    ].edge_index

    neg_edge_index = data_neg[
        "user", "rates", "item"
    ].edge_index

    if pos_edge_index.size(1) != len(pos_df):
        raise RuntimeError(
            "Positive edge count mismatch:\n"
            f"graph={pos_edge_index.size(1)}\n"
            f"csv={len(pos_df)}"
        )

    if neg_edge_index.size(1) != len(neg_df):
        raise RuntimeError(
            "Negative edge count mismatch:\n"
            f"graph={neg_edge_index.size(1)}\n"
            f"csv={len(neg_df)}"
        )

    pos_user_idx = pos_edge_index[0].long()
    pos_item_idx = pos_edge_index[1].long()

    neg_user_idx = neg_edge_index[0].long()
    neg_item_idx = neg_edge_index[1].long()

    (
        review_emb_all,
        pos_review_raw_np,
        neg_review_raw_np,
        embedding_file,
    ) = load_text_only_edge_embeddings(
        project_dir=project_dir,
        embedding_file=Path(args.edge_attr_file),
        train_df=pos_df,
        negative_df=neg_df,
    )

    text_edge_dim = int(review_emb_all.shape[1])

    print(f"Explicit edge file: {embedding_file}")
    print(f"Review embedding shape: {review_emb_all.shape}")
    print(f"Text edge dimension: {text_edge_dim}")

    if checkpoint_edge_dim != text_edge_dim:
        raise RuntimeError(
            "\nTEXT-ONLY EDGE DIMENSION MISMATCH\n"
            f"Checkpoint expects: {checkpoint_edge_dim}\n"
            f"Embedding file provides: {text_edge_dim}\n\n"
            "This stage must use the text-only checkpoint and "
            "review_embeddings_text_only.npy.\n"
            "Do NOT use old edge_features.npy from the ABSA phase."
        )

    pos_edge_attr = torch.from_numpy(
        pos_review_raw_np
    )

    data["user", "rates", "item"].edge_attr = pos_edge_attr
    data["item", "rated_by", "user"].edge_attr = pos_edge_attr

    data_h = data.to_homogeneous()

    if not hasattr(data_h, "edge_attr") or data_h.edge_attr is None:
        raise RuntimeError(
            "edge_attr missing after to_homogeneous()."
        )

    homogeneous_edge_dim = int(
        data_h.edge_attr.shape[-1]
    )

    if homogeneous_edge_dim != text_edge_dim:
        raise RuntimeError(
            f"Unexpected homogeneous edge dimension: "
            f"{homogeneous_edge_dim}; "
            f"expected {text_edge_dim}"
        )

    model = ReFINe_plus(
        num_nodes=data_h.num_nodes,
        embedding_dim=embedding_dim,
        num_layers=layers,
        num_users=num_users,
        num_items=num_items,
        edge_attr_dim=text_edge_dim,
        learnable_alpha=False,
    )

    try:
        model.load_state_dict(
            state_dict,
            strict=True,
        )
    except RuntimeError as e:
        raise RuntimeError(
            "\nCheckpoint/model mismatch.\n"
            f"checkpoint={checkpoint_path}\n"
            f"checkpoint_edge_dim={checkpoint_edge_dim}\n"
            f"current_edge_dim={text_edge_dim}\n"
        ) from e

    model.eval()

    base_ckpt = torch.load(
        base_embeddings_path,
        map_location="cpu",
    )

    if "user_emb" not in base_ckpt:
        raise RuntimeError(
            "base_embeddings file does not contain 'user_emb'."
        )

    if "item_emb" not in base_ckpt:
        raise RuntimeError(
            "base_embeddings file does not contain 'item_emb'."
        )

    base_user_emb = base_ckpt["user_emb"].float()
    base_item_emb = base_ckpt["item_emb"].float()

    if base_user_emb.shape != (
        num_users,
        embedding_dim,
    ):
        raise RuntimeError(
            f"user_emb shape mismatch: "
            f"expected={(num_users, embedding_dim)}, "
            f"got={tuple(base_user_emb.shape)}"
        )

    if base_item_emb.shape != (
        num_items,
        embedding_dim,
    ):
        raise RuntimeError(
            f"item_emb shape mismatch: "
            f"expected={(num_items, embedding_dim)}, "
            f"got={tuple(base_item_emb.shape)}"
        )

    print(
        f"Base user embedding: {tuple(base_user_emb.shape)}"
    )
    print(
        f"Base item embedding: {tuple(base_item_emb.shape)}"
    )

    with torch.no_grad():

        neg_review_raw = torch.from_numpy(
            neg_review_raw_np
        )

        pos_review_raw = torch.from_numpy(
            pos_review_raw_np
        )

        if model.edge_attr_proj is not None:
            e_r = model.compute_edge_feat_direction(
                neg_review_raw
            )

            pos_dirs = model.compute_edge_feat_direction(
                pos_review_raw
            )
        else:
            e_r = F.normalize(
                neg_review_raw,
                dim=-1,
            )

            pos_dirs = F.normalize(
                pos_review_raw,
                dim=-1,
            )

    centroid = torch.zeros(
        num_users,
        pos_dirs.size(-1),
        dtype=torch.float,
    )

    counts = torch.zeros(
        num_users,
        dtype=torch.float,
    )

    centroid.index_add_(
        0,
        pos_user_idx,
        pos_dirs,
    )

    counts.index_add_(
        0,
        pos_user_idx,
        torch.ones(
            pos_user_idx.size(0),
            dtype=torch.float,
        ),
    )

    counts = counts.clamp(
        min=1
    ).unsqueeze(-1)

    centroid = F.normalize(
        centroid / counts,
        dim=-1,
    )

    # Similarity of negative review to user's positive review centroid
    sim = (
        e_r
        * centroid[neg_user_idx]
    ).sum(-1)

    user_pos_count_arr = np.bincount(
        pos_user_idx.numpy(),
        minlength=num_users,
    )

    user_neg_count_arr = np.bincount(
        neg_user_idx.numpy(),
        minlength=num_users,
    )

    item_pop_arr = np.bincount(
        pos_item_idx.numpy(),
        minlength=num_items,
    )

    # Item average rating from positive train edges
    rating_col = (
        "rating:float"
        if "rating:float" in pos_df.columns
        else (
            "rating"
            if "rating" in pos_df.columns
            else None
        )
    )

    if rating_col is None:
        raise ValueError(
            "train_edges.csv must contain rating:float or rating."
        )

    ratings = pd.to_numeric(
        pos_df[rating_col],
        errors="coerce",
    ).fillna(0.0).to_numpy()

    if len(ratings) != len(pos_item_idx):
        raise RuntimeError(
            "Rating count does not match positive edge count."
        )

    item_rating_sum = np.zeros(
        num_items,
        dtype=np.float64,
    )

    np.add.at(
        item_rating_sum,
        pos_item_idx.numpy(),
        ratings,
    )

    item_avg_rating_arr = np.divide(
        item_rating_sum,
        np.maximum(item_pop_arr, 1),
    )

    def z(x):
        x = np.asarray(
            x,
            dtype=np.float32,
        )

        std = float(x.std())

        if std < 1e-8:
            return np.zeros_like(x)

        return (x - x.mean()) / std

    neg_users_np = neg_user_idx.numpy()
    neg_items_np = neg_item_idx.numpy()

    feats = pd.DataFrame(
        {
            "row_index": neg_df["row_index"].to_numpy(),
            "user_idx": neg_users_np,
            "item_idx": neg_items_np,

            # semantic feature
            "sim_to_positive_centroid": sim.numpy(),

            # behavioral/statistical features
            "user_pos_count": z(
                user_pos_count_arr[neg_users_np]
            ),
            "user_neg_count": z(
                user_neg_count_arr[neg_users_np]
            ),
            "item_popularity": z(
                item_pop_arr[neg_items_np]
            ),
            "item_avg_rating": z(
                item_avg_rating_arr[neg_items_np]
            ),
        }
    )

    selected_user_emb = base_user_emb[neg_user_idx]
    selected_item_emb = base_item_emb[neg_item_idx]

    er_np = e_r.numpy()
    eu_np = selected_user_emb.numpy()
    ei_np = selected_item_emb.numpy()
    
    er_cols = {
    f"e_r_{d}": er_np[:, d]
    for d in range(er_np.shape[1])
    }

    eu_cols = {
        f"e_u_{d}": eu_np[:, d]
        for d in range(eu_np.shape[1])
    }

    ei_cols = {
        f"e_i_{d}": ei_np[:, d]
        for d in range(ei_np.shape[1])
    }
    
    embedding_features = pd.DataFrame(
        {
            **er_cols,
            **eu_cols,
            **ei_cols,
        }
    )

    feats = pd.concat(
        [feats.reset_index(drop=True), embedding_features],
        axis=1,
    )
    
    feats = feats.merge(
        llm_df,
        on="row_index",
        how="left",
        validate="one_to_one",
    ).copy()

    feats["has_llm_label"] = (
        feats["negative_confidence"].notna()
    )

    if len(feats) != len(neg_df):
        raise RuntimeError(
            "Distillation dataset row count changed after LLM merge."
        )

    if feats["row_index"].duplicated().any():
        raise RuntimeError(
            "Duplicate row_index detected in final distillation dataset."
        )

    print("=" * 80)
    print(
        f"Total negative edges: {len(feats)}"
    )
    print(
        f"LLM-labeled negatives: "
        f"{int(feats['has_llm_label'].sum())}"
    )
    print(
        f"LLM label fraction: "
        f"{feats['has_llm_label'].mean():.4f}"
    )
    print(
        f"Feature columns: {len(feats.columns)}"
    )
    print("=" * 80)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    feats.to_parquet(
        output_path,
        index=False,
    )

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()