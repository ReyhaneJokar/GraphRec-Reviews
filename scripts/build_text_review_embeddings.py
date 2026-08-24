"""
Build text-only review embeddings, but ONLY for reviews that are actually
read downstream:
  - positive (train_edges.csv)  -> attached as edge_attr to the LightGCN graph
  - negative (negative_edges.csv) -> used as e_review by build_distillation_dataset.py later

Saves under BOTH:
  review_embeddings_text_only.npy  (explicit name our own scripts look for)
  review_embeddings.npy            (the fallback name data_loader.py's
                                     _load_edge_features() auto-picks when
                                     edge_features.npy is absent)
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


def build_review_text(row: pd.Series, max_chars: int) -> str:
    summary = str(row.get("review_summary", "")).strip()
    text = str(row.get("text", "")).strip()
    if summary and summary.lower() != "nan":
        combined = f"Summary: {summary}\nReview: {text}"
    else:
        combined = text
    return combined[:max_chars]


def load_sentence_transformer_with_retry(model_name: str, local_model_dir, retries: int):
    from sentence_transformers import SentenceTransformer

    source = local_model_dir if local_model_dir else model_name
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            return SentenceTransformer(source)
        except Exception as e:
            last_err = e
            wait = min(2 ** attempt, 30)
            print(f"[WARN] Failed to load model (attempt {attempt}/{retries}): {e}")
            if attempt < retries:
                print(f"Retrying in {wait}s ...")
                time.sleep(wait)
    raise RuntimeError(f"Could not load '{source}' after {retries} attempts.") from last_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", required=True, help="Folder with train_edges.csv / negative_edges.csv / etc.")
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_chars", type=int, default=1500)
    ap.add_argument("--hf_endpoint", default=None, help="Override HF hub endpoint")
    ap.add_argument("--hf_timeout", type=float, default=60.0, help="Per-request timeout in seconds (default requests timeout is 10s).")
    ap.add_argument("--local_model_dir", default=None, help="Load the model from a local folder instead of the network.")
    ap.add_argument("--load_retries", type=int, default=5)
    args = ap.parse_args()

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    hf_timeout_int = str(int(round(args.hf_timeout)))
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = hf_timeout_int
    os.environ["HF_HUB_ETAG_TIMEOUT"] = hf_timeout_int

    project_dir = Path(args.project_dir)

    train_df = pd.read_csv(project_dir / "train_edges.csv")
    neg_df = pd.read_csv(project_dir / "negative_edges.csv")
    needed = pd.concat([train_df, neg_df], ignore_index=True)
    needed = needed.drop_duplicates(subset=["row_index"]).sort_values("row_index").reset_index(drop=True)

    all_row_index_max = -1
    for fname in ["train_edges.csv", "val_edges.csv", "test_edges.csv", "negative_edges.csv", "neutral_edges.csv"]:
        fpath = project_dir / fname
        if fpath.exists():
            col = pd.read_csv(fpath, usecols=["row_index"])["row_index"]
            if len(col):
                all_row_index_max = max(all_row_index_max, int(col.max()))
    n_total = all_row_index_max + 1

    print(f"Full corpus row_index range: 0..{all_row_index_max} (n_total={n_total})")
    print(f"Encoding only {len(needed)} reviews (positive train + negative), skipping "
          f"val/test/neutral -- {n_total - len(needed)} rows stay zero-vectors "
          f"(never read by data_loader.py or build_distillation_dataset.py).")

    texts = [build_review_text(row, args.max_chars) for _, row in needed.iterrows()]

    print(f"Loading {args.model} ...")
    model = load_sentence_transformer_with_retry(args.model, args.local_model_dir, args.load_retries)

    print(f"Encoding {len(texts)} reviews ...")
    computed = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)

    emb_dim = computed.shape[1]
    full_emb = np.zeros((n_total, emb_dim), dtype=np.float32)
    full_emb[needed["row_index"].astype(int).values] = computed

    out_text_only = project_dir / "review_embeddings_text_only.npy"
    out_fallback = project_dir / "review_embeddings.npy"
    np.save(out_text_only, full_emb)
    np.save(out_fallback, full_emb)

    print("Done.")
    print("Full array shape:", full_emb.shape, f"({len(needed)} real rows, {n_total - len(needed)} zero rows)")
    print("Saved:", out_text_only)
    print("Saved:", out_fallback, "(same content -- this is the name data_loader.py auto-detects)")


if __name__ == "__main__":
    main()
