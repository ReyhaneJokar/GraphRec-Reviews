import argparse
import copy
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


SIMPLE_FEATURES = [
    "sim_to_positive_centroid",
    "user_pos_count",
    "user_neg_count",
    "item_popularity",
    "item_avg_rating",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class StudentMLP(nn.Module):
    def __init__(self, input_dim: int, hidden1: int = 16, hidden2: int = 8):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def select_features(df: pd.DataFrame, feature_set: str):
    if feature_set == "simple":
        missing = [c for c in SIMPLE_FEATURES if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing simple features: {missing}"
            )
        return SIMPLE_FEATURES

    if feature_set == "full":
        base = [
            "sim_to_positive_centroid",
            "user_pos_count",
            "user_neg_count",
            "item_popularity",
            "item_avg_rating",
        ]

        embedding_cols = [
            c for c in df.columns
            if c.startswith("e_r_")
            or c.startswith("e_u_")
            or c.startswith("e_i_")
        ]

        cols = base + sorted(embedding_cols)

        if not embedding_cols:
            raise ValueError(
                "No embedding features found for feature_set=full."
            )

        return cols

    raise ValueError(
        f"Unknown feature_set={feature_set}. Use simple or full."
    )


def train_one_fold(
    X_train,
    y_train,
    X_val,
    y_val,
    input_dim,
    seed,
    epochs,
    patience,
    lr,
    weight_decay,
):
    set_seed(seed)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    Xtr = torch.tensor(
        X_train_scaled,
        dtype=torch.float32,
    )
    ytr = torch.tensor(
        y_train,
        dtype=torch.float32,
    )

    Xva = torch.tensor(
        X_val_scaled,
        dtype=torch.float32,
    )
    yva = torch.tensor(
        y_val,
        dtype=torch.float32,
    )

    model = StudentMLP(input_dim=input_dim)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad()

        pred = model(Xtr)
        loss = nn.functional.mse_loss(pred, ytr)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        model.eval()

        with torch.no_grad():
            val_pred = model(Xva)
            val_loss = nn.functional.mse_loss(
                val_pred,
                yva,
            ).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(
                model.state_dict()
            )
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            break

    if best_state is None:
        raise RuntimeError("No best model state was saved.")

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        val_pred = model(Xva).numpy()

    return model, scaler, val_pred, best_val


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--distill_dataset", required=True,)
    parser.add_argument("--sample_frac", type=float, required=True,)
    parser.add_argument("--feature_set", choices=["simple", "full"], default="simple",)
    parser.add_argument("--n_folds", type=int, default=5,)
    parser.add_argument("--epochs", type=int, default=300,)
    parser.add_argument("--patience", type=int, default=30,)
    parser.add_argument("--lr", type=float, default=1e-3,)
    parser.add_argument("--weight_decay", type=float, default=1e-3,)
    parser.add_argument("--weight_floor", type=float, default=1.0,
        help="Final weight when negative_confidence=0, i.e. the LLM/MLP thinks "
             "this 'confirmed negative' is probably NOT a genuine dislike. "
             "Default 1.0 = treat it no worse than a generic unrated/unknown "
             "item (matches negative_sampling_probabilities' default elsewhere "
             "in main.py). Do not set below 1.0 unless you deliberately want "
             "unreliable-looking confirmed negatives penalized below unknowns.",
    )
    parser.add_argument("--weight_ceiling", type=float, default=1.5,
        help="Final weight when negative_confidence=1 (fully trusted genuine "
             "dislike). MUST match --real_neg_samp_prob of the main.py run this "
             "feeds into (--neg_confidence_weights_path), or you silently make "
             "the confidence-weighted run weaker/stronger on negative feedback "
             "than the non-distilled baseline you are comparing it against.",
    )
    parser.add_argument("--seed", type=int, default=1337,)
    parser.add_argument("--output_weights", required=True,)
    parser.add_argument("--output_model", default=None,)
    args = parser.parse_args()

    set_seed(args.seed)

    dataset_path = Path(args.distill_dataset)
    output_weights = Path(args.output_weights)

    df = pd.read_parquet(dataset_path)

    if "has_llm_label" not in df.columns:
        raise ValueError(
            "distill_dataset must contain has_llm_label."
        )

    if "negative_confidence" not in df.columns:
        raise ValueError(
            "distill_dataset must contain negative_confidence."
        )

    labeled = df[
        df["has_llm_label"] == True
    ].copy()

    labeled["negative_confidence"] = pd.to_numeric(
        labeled["negative_confidence"],
        errors="coerce",
    )

    labeled = labeled.dropna(
        subset=["negative_confidence"]
    )

    labeled["negative_confidence"] = labeled[
        "negative_confidence"
    ].clip(0.0, 1.0)

    if len(labeled) < 10:
        raise RuntimeError(
            f"Only {len(labeled)} labeled samples available. "
            "Too few for reliable K-fold training."
        )

    feature_cols = select_features(
        df,
        args.feature_set,
    )

    X_all = df[
        feature_cols
    ].astype(np.float32).to_numpy()

    X_lab = labeled[
        feature_cols
    ].astype(np.float32).to_numpy()

    y_lab = labeled[
        "negative_confidence"
    ].astype(np.float32).to_numpy()

    row_indices_labeled = labeled[
        "row_index"
    ].astype(int).to_numpy()

    print("=" * 80)
    print("Negative-confidence Student MLP")
    print("=" * 80)

    print(f"Dataset: {dataset_path}")
    print(f"Feature set: {args.feature_set}")
    print(f"Number of features: {len(feature_cols)}")
    print(f"Total negative edges: {len(df)}")
    print(f"LLM-labeled samples: {len(labeled)}")
    print(f"Requested sample fraction: {args.sample_frac:.4f}")
    print(
        f"Actual labeled fraction: "
        f"{len(labeled) / len(df):.4f}"
    )
    print(
        f"Target mean: {y_lab.mean():.6f}"
    )
    print(
        f"Target std: {y_lab.std():.6f}"
    )
    print("=" * 80)

    n_splits = min(
        args.n_folds,
        len(labeled),
    )

    if n_splits < 2:
        raise RuntimeError(
            "Need at least 2 labeled samples for K-fold."
        )

    kf = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=args.seed,
    )

    oof_pred = np.zeros(
        len(labeled),
        dtype=np.float32,
    )

    fold_models = []
    fold_scalers = []
    fold_losses = []

    for fold, (train_idx, val_idx) in enumerate(
        kf.split(X_lab)
    ):
        model, scaler, val_pred, best_val = train_one_fold(
            X_train=X_lab[train_idx],
            y_train=y_lab[train_idx],
            X_val=X_lab[val_idx],
            y_val=y_lab[val_idx],
            input_dim=X_lab.shape[1],
            seed=args.seed + fold,
            epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        oof_pred[val_idx] = val_pred

        fold_models.append(model)
        fold_scalers.append(scaler)
        fold_losses.append(best_val)

        print(
            f"fold {fold}: "
            f"best_val_mse={best_val:.6f}"
        )

    oof_mse = mean_squared_error(
        y_lab,
        oof_pred,
    )

    baseline_pred = np.full_like(
        y_lab,
        y_lab.mean(),
    )

    baseline_mse = mean_squared_error(
        y_lab,
        baseline_pred,
    )

    improvement = (
        100.0
        * (baseline_mse - oof_mse)
        / baseline_mse
    )

    print()
    print("=" * 80)
    print(f"OOF MSE: {oof_mse:.6f}")
    print(f"Mean baseline MSE: {baseline_mse:.6f}")
    print(f"Improvement over mean baseline: "f"{improvement:.2f}%")
    if improvement < 0:
        print(
            "[WARN] Negative improvement -- the model is doing WORSE than "
            "predicting the constant mean for every sample. Its per-edge "
            "predictions carry no reliable signal yet; treat the resulting "
            "weights as close to a uniform discount, not real differentiation. "
            "Consider: more labeled samples, fewer features (try --feature_set "
            "simple), or heavier regularization before trusting --feature_set full."
        )
    print(f"Fold MSE mean: "f"{np.mean(fold_losses):.6f}")
    print(f"Fold MSE std: "f"{np.std(fold_losses):.6f}")
    print("=" * 80)

    all_predictions = []

    for model, scaler in zip(
        fold_models,
        fold_scalers,
    ):
        X_all_scaled = scaler.transform(
            X_all
        )

        X_all_scaled_t = torch.tensor(
            X_all_scaled,
            dtype=torch.float32,
        )

        model.eval()

        with torch.no_grad():
            pred = model(
                X_all_scaled_t
            ).numpy()

        all_predictions.append(pred)

    all_pred = np.mean(
        np.stack(all_predictions, axis=0),
        axis=0,
    )

    all_pred = np.clip(
        all_pred,
        0.0,
        1.0,
    )

    label_map = dict(
        zip(
            row_indices_labeled,
            y_lab,
        )
    )

    final_weight = all_pred.copy()

    for i, row_index in enumerate(
        df["row_index"].astype(int)
    ):
        if row_index in label_map:
            final_weight[i] = label_map[
                row_index
            ]

    final_weight = (
        args.weight_floor
        + (args.weight_ceiling - args.weight_floor)
        * final_weight
    )

    output_weights.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output_weights,
        final_weight.astype(np.float32),
    )

    print()
    print(f"Saved weights for "f"{len(final_weight)} negative edges:")
    print(output_weights)

    print(f"mean w'={final_weight.mean():.6f}")
    print(f"std w'={final_weight.std():.6f}")

    if args.output_model:
        model_path = Path(args.output_model)
        model_path.parent.mkdir(parents=True, exist_ok=True,)

        joblib.dump(
            {
                "feature_cols": feature_cols,
                "feature_set": args.feature_set,
                "models": fold_models,
                "scalers": fold_scalers,
                "weight_floor": args.weight_floor,
                "weight_ceiling": args.weight_ceiling,
                "seed": args.seed,
                "sample_frac": args.sample_frac,
            },
            model_path,
        )

        print(f"Saved student ensemble: {model_path}")


if __name__ == "__main__":
    main()