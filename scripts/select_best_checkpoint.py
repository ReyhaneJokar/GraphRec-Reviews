import argparse
import json
import shutil
from pathlib import Path


def score(metrics: dict, metric: str, k: int) -> float:
    if metric == "recall":
        return metrics[f"Recall@{k}"]
    if metric == "ndcg":
        return metrics[f"NDCG@{k}"]
    if metric == "combined":
        return (metrics[f"Recall@{k}"] + metrics[f"NDCG@{k}"]) / 2.0
    raise ValueError(f"Unknown metric: {metric}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--metric", default="combined", choices=["recall", "ndcg", "combined"], help="Matches --early_stop_metric used during training, by default.")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--copy_to", default=None)
    args = ap.parse_args()

    result_dir = Path(args.result_dir)
    candidates = sorted(result_dir.glob(f"{args.prefix}*_best_val_metrics.json"))

    if not candidates:
        raise FileNotFoundError(
            f"No files matching {args.prefix}*_best_val_metrics.json in {result_dir}. "
            f"This file is only written by the patched main.py (val_metrics_path fix) -- "
            f"if these runs happened before the patch, re-run them, or manually grep the "
            f"'Best Validation Results' block out of the .log files instead."
        )

    print(f"Found {len(candidates)} candidate runs in {result_dir}:")
    scored = []
    for path in candidates:
        with path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        s = score(metrics, args.metric, args.k)
        run_name = path.name.replace("_best_val_metrics.json", "")
        ckpt_path = result_dir / f"{run_name}.pt"
        scored.append((s, run_name, ckpt_path, metrics))
        print(f"  {run_name}: val {args.metric}@{args.k} = {s:.4f}")

    scored.sort(key=lambda x: -x[0])
    best_score, best_run, best_ckpt, best_metrics = scored[0]

    print()
    print("=" * 60)
    print(f"Best run (by VALIDATION {args.metric}@{args.k}): {best_run}")
    print(f"Score: {best_score:.4f}")
    print(f"Checkpoint: {best_ckpt}")
    print(f"Full validation metrics: {json.dumps(best_metrics, indent=2)}")
    print("=" * 60)

    if not best_ckpt.exists():
        raise FileNotFoundError(f"Winning checkpoint file not found on disk: {best_ckpt}")

    if args.copy_to:
        dest = Path(args.copy_to)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_ckpt, dest)
        print(f"Copied winning checkpoint to: {dest}")


if __name__ == "__main__":
    main()
