from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import HeteroData


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "row_index" not in df.columns:
        df = df.copy()
        df["row_index"] = df.index.astype(int)
    return df


def _load_edge_features(project_dir: Path, n_rows: int) -> Optional[np.ndarray]:
    for fname in ("edge_features.npy", "review_embeddings.npy"):
        emb_path = project_dir / fname
        if not emb_path.exists():
            continue

        emb = np.load(emb_path)
        if emb.ndim != 2:
            raise ValueError(f"{emb_path} must be a 2D array, got shape={emb.shape}")
        if emb.shape[0] < n_rows:
            raise ValueError(
                f"{emb_path} has {emb.shape[0]} rows but input needs at least {n_rows}"
            )
        return emb

    return None


def _attach_edge_attr_from_embeddings(
    df: pd.DataFrame, embeddings: Optional[np.ndarray]
) -> Optional[torch.Tensor]:
    if embeddings is None:
        return None

    idx = df["row_index"].astype(int).values
    edge_attr = embeddings[idx]
    return torch.tensor(edge_attr, dtype=torch.float)


def _make_edge_index(df: pd.DataFrame) -> torch.Tensor:
    return torch.tensor(
        np.stack([df["user_id:token"].values, df["item_id:token"].values]),
        dtype=torch.long,
    )


def _fit_label_encoders(*dfs: pd.DataFrame) -> Tuple[LabelEncoder, LabelEncoder]:
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    all_users = pd.concat([df["user_id:token"] for df in dfs], axis=0).astype(str)
    all_items = pd.concat([df["item_id:token"] for df in dfs], axis=0).astype(str)

    user_encoder.fit(all_users.values)
    item_encoder.fit(all_items.values)

    return user_encoder, item_encoder


def _encode_ids(
    df: pd.DataFrame, user_encoder: LabelEncoder, item_encoder: LabelEncoder
) -> pd.DataFrame:
    out = df.copy()
    out["user_id:token"] = user_encoder.transform(out["user_id:token"].astype(str).values)
    out["item_id:token"] = item_encoder.transform(out["item_id:token"].astype(str).values)
    return out


def data_loading(project_dir: str, load_val_or_test: str = "val"):
    """
    Expected files in project_dir:
      - train_edges.csv
      - val_edges.csv
      - test_edges.csv
      - negative_edges.csv
      - neutral_edges.csv
      - edge_features.npy        (preferred, optional)
      - review_embeddings.npy   (optional, aligned by row_index)
    """

    project_dir = Path(project_dir)

    train_pos = _read_csv(project_dir / "train_edges.csv")
    val_pos = _read_csv(project_dir / "val_edges.csv")
    test_pos = _read_csv(project_dir / "test_edges.csv")
    neg_df = _read_csv(project_dir / "negative_edges.csv")
    neu_df = _read_csv(project_dir / "neutral_edges.csv")

    user_encoder, item_encoder = _fit_label_encoders(train_pos, val_pos, test_pos, neg_df, neu_df)

    train_pos = _encode_ids(train_pos, user_encoder, item_encoder)
    val_pos = _encode_ids(val_pos, user_encoder, item_encoder)
    test_pos = _encode_ids(test_pos, user_encoder, item_encoder)
    neg_df = _encode_ids(neg_df, user_encoder, item_encoder)
    neu_df = _encode_ids(neu_df, user_encoder, item_encoder)

    embeddings = _load_edge_features(project_dir, n_rows=max(
        train_pos["row_index"].max(),
        val_pos["row_index"].max(),
        test_pos["row_index"].max(),
        neg_df["row_index"].max(),
        neu_df["row_index"].max(),
    ) + 1)

    data = HeteroData()
    data_neg = HeteroData()
    data_neutral = HeteroData()

    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)

    data["user"].num_nodes = num_users
    data["item"].num_nodes = num_items

    data_neg["user"].num_nodes = num_users
    data_neg["item"].num_nodes = num_items

    data_neutral["user"].num_nodes = num_users
    data_neutral["item"].num_nodes = num_items

    # Positive train graph (graph branch)
    pos_edge_index = _make_edge_index(train_pos)
    pos_edge_rate = torch.tensor(train_pos["rating:float"].values, dtype=torch.float)
    pos_edge_attr = _attach_edge_attr_from_embeddings(train_pos, embeddings)

    data["user", "rates", "item"].edge_index = pos_edge_index
    data["user", "rates", "item"].edge_rate = pos_edge_rate
    if pos_edge_attr is not None:
        data["user", "rates", "item"].edge_attr = pos_edge_attr

    data["item", "rated_by", "user"].edge_index = pos_edge_index.flip([0])
    data["item", "rated_by", "user"].edge_rate = pos_edge_rate
    if pos_edge_attr is not None:
        data["item", "rated_by", "user"].edge_attr = pos_edge_attr

    # Negative branch for autoencoder
    neg_edge_index = _make_edge_index(neg_df)
    data_neg["user", "rates", "item"].edge_index = neg_edge_index
    data_neg["item", "rated_by", "user"].edge_index = neg_edge_index.flip([0])

    # Neutral branch (kept for completeness / later ablation)
    neu_edge_index = _make_edge_index(neu_df)
    data_neutral["user", "rates", "item"].edge_index = neu_edge_index
    data_neutral["item", "rated_by", "user"].edge_index = neu_edge_index.flip([0])

    # Validation / test: positive-only labels
    if load_val_or_test == "val":
        edge_label_index = _make_edge_index(val_pos)
        data["user", "rates", "item"].edge_label_index = edge_label_index
    elif load_val_or_test == "test":
        edge_label_index = _make_edge_index(test_pos)
        data["user", "rates", "item"].edge_label_index = edge_label_index
    else:
        raise ValueError("load_val_or_test must be 'val' or 'test'")

    return data, data_neg, data_neutral