import argparse
import json
import re
from pathlib import Path

BLOCK_RE = re.compile(
    r"Early stopping at epoch \d+\. Best Validation Results:\s*\n((?:.*\n)+?)(?:\n|$)"
)
LINE_RE = re.compile(
    r"Precision@(\d+):\s*([\d.]+),\s*Recall@(\d+):\s*([\d.]+),\s*NDCG@(\d+):\s*([\d.]+)"
)


def parse_log(log_path: Path):
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    m = BLOCK_RE.search(text)
    if not m:
        return None
    metrics = {}
    for line in m.group(1).splitlines():
        lm = LINE_RE.search(line)
        if lm:
            k = lm.group(1)
            metrics[f"Precision@{k}"] = float(lm.group(2))
            metrics[f"Recall@{k}"] = float(lm.group(4))
            metrics[f"NDCG@{k}"] = float(lm.group(6))
    return metrics or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", required=True)
    ap.add_argument("--result_dir", required=True)
    args = ap.parse_args()
    logs_dir, result_dir = Path(args.logs_dir), Path(args.result_dir)

    if not logs_dir.exists():
        raise FileNotFoundError(f"logs_dir not found: {logs_dir}")
    if not result_dir.exists():
        raise FileNotFoundError(f"result_dir not found: {result_dir}")

    recovered, missing = [], []
    for log_path in sorted(logs_dir.glob("edge_aware_seed*.log")):
        run_name = log_path.stem
        metrics = parse_log(log_path)
        if metrics:
            out_path = result_dir / f"{run_name}_best_val_metrics.json"
            out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
            recovered.append(run_name)
        else:
            missing.append(run_name)

    print(f"Retrieved: {len(recovered)} -> {recovered}")
    print(f"Remainder: {len(missing)} -> {missing}")


if __name__ == "__main__":
    main()
