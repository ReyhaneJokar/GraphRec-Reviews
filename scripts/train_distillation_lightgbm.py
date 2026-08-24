import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

SIMPLE_FEATURES = [
    "sim_to_positive_centroid",
    "user_pos_count",
    "user_neg_count",
    "item_popularity",
    "item_avg_rating",
]


def select_features(df: pd.DataFrame, feature_set: str):
    if feature_set == "simple":
        missing = [c for c in SIMPLE_FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"Missing simple features: {missing}")
        return SIMPLE_FEATURES

    if feature_set == "full":
        base = SIMPLE_FEATURES
        embedding_cols = [
            c for c in df.columns
            if c.startswith("e_r_") or c.startswith("e_u_") or c.startswith("e_i_")
        ]
        if not embedding_cols:
            raise ValueError("No embedding features found for feature_set=full.")
        return base + sorted(embedding_cols)

    raise ValueError(f"Unknown feature_set={feature_set}. Use simple or full.")


def main():
    import lightgbm as lgb

    parser = argparse.ArgumentParser()
    parser.add_argument("--distill_dataset", required=True)
    parser.add_argument("--sample_frac", type=float, required=True)
    parser.add_argument("--feature_set", choices=["simple", "full"], default="simple")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--num_leaves", type=int, default=7)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=0.03)
    parser.add_argument("--min_child_samples", type=int, default=20)
    parser.add_argument("--reg_alpha", type=float, default=1.0)
    parser.add_argument("--reg_lambda", type=float, default=1.0)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--early_stopping_rounds", type=int, default=30)
    parser.add_argument("--weight_floor", type=float, default=1.0, help="Final weight when negative_confidence=0. Default 1.0 = no worse than a generic unknown item.")
    parser.add_argument("--weight_ceiling", type=float, default=1.5, help="Final weight when negative_confidence=1. MUST match --real_neg_samp_prob of the main.py run this feeds into.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output_weights", required=True)
    parser.add_argument("--output_model", default=None)
    args = parser.parse_args()

    dataset_path = Path(args.distill_dataset)
    output_weights = Path(args.output_weights)

    df = pd.read_parquet(dataset_path)
    for col in ("has_llm_label", "negative_confidence", "row_index"):
        if col not in df.columns:
            raise ValueError(f"distill_dataset must contain {col}.")

    labeled = df[df["has_llm_label"] == True].copy()
    labeled["negative_confidence"] = pd.to_numeric(labeled["negative_confidence"], errors="coerce")
    labeled = labeled.dropna(subset=["negative_confidence"])
    labeled["negative_confidence"] = labeled["negative_confidence"].clip(0.0, 1.0)

    if len(labeled) < 10:
        raise RuntimeError(f"Only {len(labeled)} labeled samples available. Too few for K-fold.")

    feature_cols = select_features(df, args.feature_set)
    X_all = df[feature_cols].astype(np.float32).to_numpy()
    X_lab = labeled[feature_cols].astype(np.float32).to_numpy()
    y_lab = labeled["negative_confidence"].astype(np.float32).to_numpy()
    row_indices_labeled = labeled["row_index"].astype(int).to_numpy()

    print("=" * 80)
    print("Negative-confidence Student LightGBM")
    print("=" * 80)
    print(f"Dataset: {dataset_path}")
    print(f"Feature set: {args.feature_set}")
    print(f"Number of features: {len(feature_cols)}")
    print(f"Total negative edges: {len(df)}")
    print(f"LLM-labeled samples: {len(labeled)}")
    print(f"Actual labeled fraction: {len(labeled) / len(df):.4f}")
    print(f"Target mean: {y_lab.mean():.6f}")
    print(f"Target std: {y_lab.std():.6f}")
    print("=" * 80)

    n_splits = min(args.n_folds, len(labeled))
    if n_splits < 2:
        raise RuntimeError("Need at least 2 labeled samples for K-fold.")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    oof_pred = np.zeros(len(labeled), dtype=np.float32)
    fold_models = []
    fold_losses = []
    importances = np.zeros(len(feature_cols), dtype=np.float64)

    lgb_params = dict(
        n_estimators=args.n_estimators,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        min_child_samples=args.min_child_samples,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        verbosity=-1,
    )

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_lab)):
        model = lgb.LGBMRegressor(**lgb_params, random_state=args.seed + fold)
        model.fit(
            X_lab[train_idx], y_lab[train_idx],
            eval_set=[(X_lab[val_idx], y_lab[val_idx])],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(args.early_stopping_rounds, verbose=False)],
        )
        val_pred = np.clip(model.predict(X_lab[val_idx]), 0.0, 1.0)
        oof_pred[val_idx] = val_pred
        fold_mse = mean_squared_error(y_lab[val_idx], val_pred)
        fold_models.append(model)
        fold_losses.append(fold_mse)
        importances += model.feature_importances_ / n_splits
        print(f"fold {fold}: best_iteration={model.best_iteration_}, best_val_mse={fold_mse:.6f}")

    oof_mse = mean_squared_error(y_lab, oof_pred)
    baseline_pred = np.full_like(y_lab, y_lab.mean())
    baseline_mse = mean_squared_error(y_lab, baseline_pred)
    improvement = 100.0 * (baseline_mse - oof_mse) / baseline_mse

    print()
    print("=" * 80)
    print(f"OOF MSE: {oof_mse:.6f}")
    print(f"Mean baseline MSE: {baseline_mse:.6f}")
    print(f"Improvement over mean baseline: {improvement:.2f}%")
    if improvement < 0:
        print("[WARN] Still worse than predicting the constant mean. This points "
              "toward a genuine data-size bottleneck rather than a "
              "model-class issue -- consider labeling a larger --sample_frac "
              "before trusting per-edge weights from either model.")
    else:
        print("[OK] Beats the constant-mean baseline -- LightGBM is picking up "
              "real signal the MLP could not with the same number of labels. The MLP's "
              "failure was a model-class/optimization issue, not purely a data-"
              "size ceiling.")
    print(f"Fold MSE mean: {np.mean(fold_losses):.6f}")
    print(f"Fold MSE std: {np.std(fold_losses):.6f}")
    print("=" * 80)

    print("\nTop feature importances (avg split gain across folds):")
    order = np.argsort(-importances)
    for i in order[:15]:
        print(f"  {feature_cols[i]:30s} {importances[i]:10.2f}")

    all_predictions = np.stack(
        [np.clip(m.predict(X_all), 0.0, 1.0) for m in fold_models], axis=0
    )
    all_pred = all_predictions.mean(axis=0)

    label_map = dict(zip(row_indices_labeled, y_lab))
    final_weight = all_pred.copy()
    for i, row_index in enumerate(df["row_index"].astype(int)):
        if row_index in label_map:
            final_weight[i] = label_map[row_index]

    final_weight = args.weight_floor + (args.weight_ceiling - args.weight_floor) * final_weight

    output_weights.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_weights, final_weight.astype(np.float32))

    print()
    print(f"Saved weights for {len(final_weight)} negative edges:")
    print(output_weights)
    print(f"mean w'={final_weight.mean():.6f}")
    print(f"std w'={final_weight.std():.6f}")

    if args.output_model:
        import joblib
        model_path = Path(args.output_model)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "feature_cols": feature_cols,
                "feature_set": args.feature_set,
                "models": fold_models,
                "weight_floor": args.weight_floor,
                "weight_ceiling": args.weight_ceiling,
                "seed": args.seed,
            },
            model_path,
        )
        print(f"Saved student ensemble: {model_path}")


if __name__ == "__main__":
    main()
