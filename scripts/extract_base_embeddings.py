import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

import data_loader
from model import ReFINe_plus


def infer_checkpoint_edge_dim(state_dict):
    """
    Infer the edge feature input dimension directly from checkpoint
    without constructing the model first.

    Supports the current architecture where:
      edge_attr_proj.0 = LayerNorm(edge_attr_dim)
      edge_attr_proj.1 = Linear(edge_attr_dim, hidden)

    Also supports an older architecture where:
      edge_attr_proj.0 = Linear(hidden, edge_attr_dim)
    """

    # Current architecture:
    # LayerNorm(edge_attr_dim).weight -> [edge_attr_dim]
    key = "edge_attr_proj.0.weight"
    if key in state_dict:
        shape = tuple(state_dict[key].shape)

        if len(shape) == 1:
            return int(shape[0])

        # Older Linear(hidden, edge_attr_dim)
        if len(shape) == 2:
            return int(shape[1])

    # Fallback: inspect first 2D edge projection layer.
    candidates = [
        "edge_attr_proj.1.weight",
        "edge_attr_proj.0.weight",
    ]

    for key in candidates:
        if key not in state_dict:
            continue

        shape = tuple(state_dict[key].shape)

        if len(shape) == 2:
            return int(shape[1])

    return 0


def infer_checkpoint_embedding_dim(state_dict):
    key = "embedding.weight"

    if key not in state_dict:
        raise RuntimeError(
            "Cannot infer embedding dimension: "
            "'embedding.weight' not found in checkpoint."
        )

    shape = tuple(state_dict[key].shape)

    if len(shape) != 2:
        raise RuntimeError(
            f"Unexpected embedding.weight shape: {shape}"
        )

    return int(shape[1])


def infer_checkpoint_num_layers(state_dict):
    """
    Count conv layers from checkpoint parameter names.
    """
    layer_ids = set()

    for key in state_dict.keys():
        if key.startswith("convs."):
            parts = key.split(".")
            if len(parts) >= 2 and parts[1].isdigit():
                layer_ids.add(int(parts[1]))

    if not layer_ids:
        raise RuntimeError(
            "Could not infer number of GNN layers from checkpoint."
        )

    return max(layer_ids) + 1


def load_explicit_text_edge_features(project_dir: Path, embedding_file: Path):
    """
    Load text-only review embeddings explicitly.

    We do NOT use data_loader._load_edge_features(), because it prefers
    edge_features.npy, which belongs to the old ABSA experiment.

    The embedding rows are aligned using row_index from train_edges.csv.
    """
    if not embedding_file.is_absolute():
        embedding_file = project_dir / embedding_file

    if not embedding_file.exists():
        raise FileNotFoundError(
            f"Explicit edge embedding file not found:\n{embedding_file}"
        )

    emb = np.load(embedding_file)

    if emb.ndim != 2:
        raise ValueError(
            f"Embedding file must be 2D, got shape={emb.shape}"
        )

    train_csv = project_dir / "train_edges.csv"

    if not train_csv.exists():
        raise FileNotFoundError(
            f"Missing train_edges.csv:\n{train_csv}"
        )

    train_df = pd.read_csv(train_csv)

    if "row_index" not in train_df.columns:
        raise ValueError(
            f"{train_csv} must contain row_index."
        )

    row_indices = train_df["row_index"].astype(int).to_numpy()

    if len(row_indices) == 0:
        raise ValueError("train_edges.csv contains no training edges.")

    max_row = int(row_indices.max())

    if emb.shape[0] <= max_row:
        raise ValueError(
            f"Embedding file has {emb.shape[0]} rows, "
            f"but train_edges.csv requires row_index up to {max_row}."
        )

    train_edge_attr = emb[row_indices].astype(np.float32, copy=False)

    if train_edge_attr.shape[0] != len(train_df):
        raise RuntimeError(
            "Number of selected edge features does not match "
            "number of train edges."
        )

    return train_edge_attr, emb


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--project_dir", required=True,)
    ap.add_argument("--checkpoint", required=True,)
    ap.add_argument("--edge_attr_file", required=True, help="Explicit text-only review embedding file, e.g. review_embeddings_text_only.npy",)
    ap.add_argument("--output", required=True,)
    ap.add_argument("--embedding_dim", type=int, default=None, help="Optional. If omitted, inferred from checkpoint.",)
    ap.add_argument("--layers", type=int, default=None, help="Optional. If omitted, inferred from checkpoint.",)

    args = ap.parse_args()

    project_dir = Path(args.project_dir)
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint_path}"
        )

    print("=" * 80)
    print("Base embedding extraction")
    print("=" * 80)

    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    checkpoint_edge_dim = infer_checkpoint_edge_dim(state_dict)
    checkpoint_embedding_dim = infer_checkpoint_embedding_dim(state_dict)
    checkpoint_layers = infer_checkpoint_num_layers(state_dict)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Checkpoint embedding_dim: {checkpoint_embedding_dim}")
    print(f"Checkpoint layers: {checkpoint_layers}")
    print(f"Checkpoint edge_attr_dim: {checkpoint_edge_dim}")

    train_edge_attr, all_embeddings = load_explicit_text_edge_features(
        project_dir=project_dir,
        embedding_file=Path(args.edge_attr_file),
    )

    selected_edge_dim = int(train_edge_attr.shape[1])

    print(f"Explicit edge feature file: {args.edge_attr_file}")
    print(f"All embedding shape: {all_embeddings.shape}")
    print(f"Selected train edge_attr shape: {train_edge_attr.shape}")

    if checkpoint_edge_dim != 0 and selected_edge_dim != checkpoint_edge_dim:
        raise RuntimeError(
            "\nEDGE FEATURE DIMENSION MISMATCH\n"
            f"Checkpoint expects: {checkpoint_edge_dim}\n"
            f"Selected file provides: {selected_edge_dim}\n\n"
            "This usually means you selected a checkpoint from a different "
            "experiment.\n"
            "For this project phase you must use the text-only checkpoint "
            "with review_embeddings_text_only.npy."
        )

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

    if embedding_dim != checkpoint_embedding_dim:
        raise RuntimeError(
            f"Embedding dimension mismatch: "
            f"checkpoint={checkpoint_embedding_dim}, "
            f"requested={embedding_dim}"
        )

    if layers != checkpoint_layers:
        raise RuntimeError(
            f"Layer mismatch: "
            f"checkpoint={checkpoint_layers}, "
            f"requested={layers}"
        )

    data, _, _ = data_loader.data_loading(
        str(project_dir),
        load_val_or_test="val",
    )

    num_users = data["user"].num_nodes
    num_items = data["item"].num_nodes

    print(f"Users: {num_users}")
    print(f"Items: {num_items}")

    expected_train_edges = data["user", "rates", "item"].edge_index.shape[1]

    if train_edge_attr.shape[0] != expected_train_edges:
        raise RuntimeError(
            f"Train edge count mismatch: "
            f"CSV/features={train_edge_attr.shape[0]}, "
            f"graph={expected_train_edges}"
        )

    data["user", "rates", "item"].edge_attr = torch.from_numpy(
        train_edge_attr
    )

    data["item", "rated_by", "user"].edge_attr = torch.from_numpy(
        train_edge_attr
    )

    data_h = data.to_homogeneous()

    if not hasattr(data_h, "edge_attr") or data_h.edge_attr is None:
        raise RuntimeError(
            "edge_attr disappeared during to_homogeneous()."
        )

    actual_homogeneous_edge_dim = int(data_h.edge_attr.shape[1])

    if actual_homogeneous_edge_dim != selected_edge_dim:
        raise RuntimeError(
            f"Unexpected homogeneous edge_attr dimension: "
            f"{actual_homogeneous_edge_dim} "
            f"(expected {selected_edge_dim})"
        )

    model = ReFINe_plus(
        num_nodes=data_h.num_nodes,
        embedding_dim=embedding_dim,
        num_layers=layers,
        num_users=num_users,
        num_items=num_items,
        edge_attr_dim=selected_edge_dim,
        learnable_alpha=False,
    )

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        raise RuntimeError(
            "\nCheckpoint architecture still does not match the model.\n"
            "The most common cause is using a checkpoint from a different "
            "experiment/configuration.\n\n"
            f"checkpoint={checkpoint_path}\n"
            f"edge_attr_file={args.edge_attr_file}\n"
            f"checkpoint_edge_dim={checkpoint_edge_dim}\n"
            f"selected_edge_dim={selected_edge_dim}\n"
        ) from e

    model.eval()

    with torch.no_grad():
        out = model.get_embedding(
            data_h.edge_index,
            edge_attr=data_h.edge_attr,
        )

    user_emb = out[:num_users].cpu()
    item_emb = out[num_users:].cpu()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "user_emb": user_emb,
            "item_emb": item_emb,
            "num_users": num_users,
            "num_items": num_items,
            "edge_attr_dim": selected_edge_dim,
            "source_edge_attr_file": str(args.edge_attr_file),
            "checkpoint": str(checkpoint_path),
        },
        output_path,
    )

    print("=" * 80)
    print("Extraction completed successfully")
    print("=" * 80)
    print(f"Saved: {output_path}")
    print(f"user_emb: {tuple(user_emb.shape)}")
    print(f"item_emb: {tuple(item_emb.shape)}")
    print(f"edge_attr_dim: {selected_edge_dim}")
    print(f"edge source: {args.edge_attr_file}")


if __name__ == "__main__":
    main()