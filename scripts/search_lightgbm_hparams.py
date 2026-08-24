"""
Small random hyperparameter search for the LightGBM negative-confidence
distiller, selected ONLY by 5-fold OOF improvement over the constant-mean
baseline -- never by downstream Recall@20/NDCG@20. This ordering matters:
pick and LOCK the config here first, then run train_distillation_lightgbm.py
once with the winning values, then run main.py once for the official
seed-comparison. Never loop back and re-pick hyperparameters after seeing
main.py's test metrics -- that turns this into test-set overfitting
(the exact p-hacking risk flagged earlier for --weight_floor/--weight_ceiling).
"""
import argparse
import random
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


def select_features(df, feature_set):
    if feature_set == "simple":
        return SIMPLE_FEATURES
    base = SIMPLE_FEATURES
    embedding_cols = [c for c in df.columns if c.startswith(("e_r_", "e_u_", "e_i_"))]
    if not embedding_cols:
        raise ValueError("No embedding features found for feature_set=full.")
    return base + sorted(embedding_cols)


def oof_improvement(X, y, params, n_folds, seed):
    import lightgbm as lgb

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_pred = np.zeros(len(y), dtype=np.float32)
    for train_idx, val_idx in kf.split(X):
        model = lgb.LGBMRegressor(**params, random_state=seed)
        model.fit(
            X[train_idx], y[train_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        oof_pred[val_idx] = np.clip(model.predict(X[val_idx]), 0.0, 1.0)
    oof_mse = mean_squared_error(y, oof_pred)
    baseline_mse = mean_squared_error(y, np.full_like(y, y.mean()))
    return 100.0 * (baseline_mse - oof_mse) / baseline_mse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distill_dataset", required=True)
    ap.add_argument("--feature_set", choices=["simple", "full"], default="simple")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--n_trials", type=int, default=25)
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()

    df = pd.read_parquet(args.distill_dataset)
    labeled = df[df["has_llm_label"] == True].copy()
    labeled["negative_confidence"] = pd.to_numeric(labeled["negative_confidence"], errors="coerce")
    labeled = labeled.dropna(subset=["negative_confidence"])
    labeled["negative_confidence"] = labeled["negative_confidence"].clip(0.0, 1.0)

    feature_cols = select_features(df, args.feature_set)
    X = labeled[feature_cols].astype(np.float32).to_numpy()
    y = labeled["negative_confidence"].astype(np.float32).to_numpy()

    print(f"Searching {args.n_trials} configs on {len(y)} labeled samples, "
          f"feature_set={args.feature_set} ({len(feature_cols)} features)")
    print("=" * 80)

    rng = random.Random(args.seed)
    grid = {
        "num_leaves": [3, 5, 7, 11, 15],
        "max_depth": [2, 3, 4, -1],
        "min_child_samples": [10, 15, 20, 30, 40],
        "reg_alpha": [0.0, 0.1, 0.5, 1.0, 2.0],
        "reg_lambda": [0.0, 0.1, 0.5, 1.0, 2.0],
        "learning_rate": [0.02, 0.03, 0.05, 0.08],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 1.0],
        "n_estimators": [200, 300, 500],
    }

    results = []
    for trial in range(args.n_trials):
        params = {k: rng.choice(v) for k, v in grid.items()}
        params["verbosity"] = -1
        try:
            improvement = oof_improvement(X, y, params, args.n_folds, args.seed + trial)
        except Exception as e:
            print(f"trial {trial}: FAILED ({e})")
            continue
        results.append((improvement, params))
        print(f"trial {trial:2d}: OOF improvement = {improvement:+.2f}%  "
              f"(leaves={params['num_leaves']}, depth={params['max_depth']}, "
              f"min_child={params['min_child_samples']}, lr={params['learning_rate']}, "
              f"reg_a={params['reg_alpha']}, reg_l={params['reg_lambda']}, "
              f"subsample={params['subsample']}, colsample={params['colsample_bytree']}, "
              f"n_est={params['n_estimators']})")

    results.sort(key=lambda r: -r[0])
    print("=" * 80)
    print("Best config (by OOF improvement only -- do NOT re-run this search "
          "after seeing main.py Recall@20):")
    best_improvement, best_params = results[0]
    print(f"OOF improvement: {best_improvement:+.2f}%")
    for k, v in best_params.items():
        if k != "verbosity":
            print(f"  --{k} {v}")
    print()


if __name__ == "__main__":
    main()