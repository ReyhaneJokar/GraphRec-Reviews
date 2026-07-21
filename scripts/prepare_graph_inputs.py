import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd


def safe_value(x: Any) -> Any:
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:
            pass
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    return x


def load_sentiment_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "row_index" not in df.columns:
        df = df.copy()
        df["row_index"] = df.index.astype(int)
    df["row_index"] = df["row_index"].astype(int)
    return df


def load_absa_jsonl(path: Path) -> pd.DataFrame:
    records = []
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            row_index = rec.get("row_index")
            if row_index is None:
                continue

            absa = rec.get("absa") or {}
            if not isinstance(absa, dict):
                absa = {}
            aspects = absa.get("aspects", [])
            if not isinstance(aspects, list):
                aspects = []

            clean_aspects = []
            aspect_tokens = []
            for a in aspects:
                if not isinstance(a, dict):
                    continue
                aspect = str(a.get("aspect", "")).strip()
                if not aspect:
                    continue
                sentiment = str(a.get("sentiment", "neutral")).strip().lower()
                score = a.get("score", 0)
                confidence = a.get("confidence", 0.0)
                try:
                    confidence = float(confidence)
                except Exception:
                    confidence = 0.0

                clean_aspects.append(
                    {
                        "aspect": aspect,
                        "sentiment": sentiment,
                        "score": score,
                        "confidence": confidence,
                    }
                )
                aspect_tokens.append(f"{aspect}:{sentiment}:{score}:{confidence:.3f}")

            records.append(
                {
                    "row_index": int(row_index),
                    "absa_overall_sentiment": absa.get("overall_sentiment"),
                    "absa_overall_score": absa.get("overall_score"),
                    "absa_aspects_json": json.dumps(clean_aspects, ensure_ascii=False),
                    "absa_aspects_compact": "|".join(aspect_tokens),
                    "absa_raw": json.dumps(absa, ensure_ascii=False),
                }
            )

    absa_df = pd.DataFrame(records)
    if not absa_df.empty:
        absa_df["row_index"] = absa_df["row_index"].astype(int)
    return absa_df


def build_review_text(row: pd.Series) -> str:
    summary = str(row.get("review_summary", "")).strip()
    text = str(row.get("text", "")).strip()
    if summary and summary.lower() != "nan":
        return f"Summary: {summary}\nReview: {text}"
    return text


def make_user_item_maps(df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int]]:
    users = sorted(str(u) for u in df["user_id"].dropna().astype(str).unique())
    items = sorted(str(i) for i in df["item_id"].dropna().astype(str).unique())
    user_map = {u: idx for idx, u in enumerate(users)}
    item_map = {i: idx for idx, i in enumerate(items)}
    return user_map, item_map


def chronological_split(df: pd.DataFrame, user_col: str = "user_id", time_col: str = "timestamp") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Leave-last-one/two-out per user.
    1 interaction -> train
    2 interactions -> train + test
    >=3 interactions -> train + val + test
    """
    train_parts = []
    val_parts = []
    test_parts = []

    work = df.copy()
    if time_col not in work.columns:
        work[time_col] = np.arange(len(work))

    work[time_col] = pd.to_numeric(work[time_col], errors="coerce")
    work["__sort_time__"] = work[time_col].fillna(work["row_index"])
    work[user_col] = work[user_col].astype(str)

    for _, g in work.groupby(user_col, sort=False):
        g = g.sort_values(["__sort_time__", "row_index"]).reset_index(drop=True)
        n = len(g)
        if n == 1:
            train_parts.append(g)
        elif n == 2:
            train_parts.append(g.iloc[:1])
            test_parts.append(g.iloc[1:])
        else:
            train_parts.append(g.iloc[:-2])
            val_parts.append(g.iloc[-2:-1])
            test_parts.append(g.iloc[-1:])

    def _concat(parts):
        if not parts:
            return pd.DataFrame(columns=df.columns)
        out = pd.concat(parts, ignore_index=True)
        if "__sort_time__" in out.columns:
            out = out.drop(columns=["__sort_time__"])
        return out

    return _concat(train_parts), _concat(val_parts), _concat(test_parts)


def try_make_embeddings(texts: List[str], model_name: str, batch_size: int = 32) -> Optional[np.ndarray]:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        print(f"[WARN] sentence-transformers is not available: {e}")
        return None

    try:
        model = SentenceTransformer(model_name)
        emb = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return np.asarray(emb, dtype=np.float32)
    except Exception as e:
        print(f"[WARN] embedding generation failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentiment_csv", required=True, help="Full sentiment CSV produced by sentiment.py")
    parser.add_argument("--absa_jsonl", required=True, help="ABSA JSONL produced for positive reviews")
    parser.add_argument("--output_dir", required=True, help="Directory for graph-ready artifacts")
    parser.add_argument("--embed_model", default="", help="Optional sentence-transformers model name, e.g. all-MiniLM-L6-v2")
    parser.add_argument("--embed_batch_size", type=int, default=32)
    parser.add_argument("--max_review_chars", type=int, default=1500)
    args = parser.parse_args()

    sentiment_path = Path(args.sentiment_csv)
    absa_path = Path(args.absa_jsonl)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sent_df = load_sentiment_csv(sentiment_path)
    absa_df = load_absa_jsonl(absa_path)

    df = sent_df.merge(absa_df, on="row_index", how="left")

    for col, default in [
        ("absa_overall_sentiment", None),
        ("absa_overall_score", None),
        ("absa_aspects_json", "[]"),
        ("absa_aspects_compact", ""),
    ]:
        if col not in df.columns:
            df[col] = default

    df["review_text"] = df.apply(build_review_text, axis=1)
    df["review_text"] = df["review_text"].astype(str).str.slice(0, args.max_review_chars)

    # Normalize fields that later code will consume.
    for col in ["user_id", "item_id", "sentiment", "review_summary", "text", "evidence", "error"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    user_map, item_map = make_user_item_maps(df)
    df["user_idx"] = df["user_id"].astype(str).map(user_map)
    df["item_idx"] = df["item_id"].astype(str).map(item_map)

    df["is_positive"] = df["sentiment"].astype(str).str.lower().eq("positive")
    df["is_negative"] = df["sentiment"].astype(str).str.lower().eq("negative")
    df["is_neutral"] = df["sentiment"].astype(str).str.lower().eq("neutral")

    merged_path = out_dir / "all_edges_merged.csv"
    df.to_csv(merged_path, index=False, encoding="utf-8-sig")

    for label, mask in [("positive", df["is_positive"]), ("negative", df["is_negative"]), ("neutral", df["is_neutral"])]:
        part = df[mask].copy()
        part.to_csv(out_dir / f"{label}_edges.csv", index=False, encoding="utf-8-sig")

    pos_cols = [
        "row_index", "user_id", "item_id", "user_idx", "item_idx",
        "rating", "timestamp", "sentiment", "score", "confidence",
        "review_summary", "text", "review_text",
        "absa_overall_sentiment", "absa_overall_score",
        "absa_aspects_json", "absa_aspects_compact", "evidence",
    ]
    pos_cols = [c for c in pos_cols if c in df.columns]
    positive_absa = df[df["is_positive"]].copy()[pos_cols]
    positive_absa.to_csv(out_dir / "positive_absa_edges.csv", index=False, encoding="utf-8-sig")

    aspect_counter = Counter()
    aspect_instances = []
    for _, row in positive_absa.iterrows():
        aspects_raw = row.get("absa_aspects_json", "[]")
        try:
            aspects = json.loads(aspects_raw) if isinstance(aspects_raw, str) else []
        except Exception:
            aspects = []
        if not isinstance(aspects, list):
            aspects = []
        for a in aspects:
            if not isinstance(a, dict):
                continue
            aspect = str(a.get("aspect", "")).strip().lower()
            if not aspect:
                continue
            aspect_counter[aspect] += 1
            aspect_instances.append(
                {
                    "row_index": int(row["row_index"]),
                    "aspect": aspect,
                    "sentiment": str(a.get("sentiment", "neutral")).lower(),
                    "score": a.get("score", 0),
                    "confidence": a.get("confidence", 0.0),
                }
            )

    aspect_vocab = {aspect: idx for idx, (aspect, _) in enumerate(aspect_counter.most_common())}
    with (out_dir / "aspect_vocab.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "num_aspects": len(aspect_vocab),
                "aspect_to_id": aspect_vocab,
                "counts": dict(aspect_counter),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    pd.DataFrame(aspect_instances).to_csv(out_dir / "positive_aspect_instances.csv", index=False, encoding="utf-8-sig")

    train_df, val_df, test_df = chronological_split(df)
    train_df.to_csv(out_dir / "train_edges.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(out_dir / "val_edges.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(out_dir / "test_edges.csv", index=False, encoding="utf-8-sig")

    embeddings = None
    if args.embed_model:
        embeddings = try_make_embeddings(
            df["review_text"].fillna("").astype(str).tolist(),
            model_name=args.embed_model,
            batch_size=args.embed_batch_size,
        )
        if embeddings is not None:
            np.save(out_dir / "review_embeddings.npy", embeddings)
            df["embedding_available"] = True
        else:
            df["embedding_available"] = False
    else:
        df["embedding_available"] = False

    def aspect_ids_from_json(js: str) -> str:
        try:
            aspects = json.loads(js) if isinstance(js, str) else []
        except Exception:
            return "[]"
        if not isinstance(aspects, list):
            return "[]"
        ids = []
        for a in aspects:
            if not isinstance(a, dict):
                continue
            aspect = str(a.get("aspect", "")).strip().lower()
            if aspect in aspect_vocab:
                ids.append(aspect_vocab[aspect])
        return json.dumps(ids, ensure_ascii=False)

    df["aspect_ids"] = df["absa_aspects_json"].fillna("[]").apply(aspect_ids_from_json)
    df["aspect_count"] = df["absa_aspects_json"].fillna("[]").apply(
        lambda s: len(json.loads(s)) if isinstance(s, str) and s else 0
    )

    if embeddings is not None:
        df["embedding_row_idx"] = np.arange(len(df), dtype=int)

    graph_ready_path = out_dir / "graph_ready_edges.csv"
    df.to_csv(graph_ready_path, index=False, encoding="utf-8-sig")

    stats = {
        "total_reviews": int(len(df)),
        "positive_reviews": int(df["is_positive"].sum()),
        "negative_reviews": int(df["is_negative"].sum()),
        "neutral_reviews": int(df["is_neutral"].sum()),
        "num_users": int(df["user_idx"].nunique()),
        "num_items": int(df["item_idx"].nunique()),
        "num_aspects": int(len(aspect_vocab)),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "embeddings_generated": bool(embeddings is not None),
        "embed_model": args.embed_model if args.embed_model else None,
    }
    with (out_dir / "graph_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    with (out_dir / "user_map.json").open("w", encoding="utf-8") as f:
        json.dump(user_map, f, ensure_ascii=False, indent=2)
    with (out_dir / "item_map.json").open("w", encoding="utf-8") as f:
        json.dump(item_map, f, ensure_ascii=False, indent=2)

    print("Done.")
    print("Saved:", merged_path)
    print("Saved:", graph_ready_path)
    print("Saved:", out_dir / "positive_absa_edges.csv")
    print("Saved:", out_dir / "train_edges.csv")
    print("Saved:", out_dir / "val_edges.csv")
    print("Saved:", out_dir / "test_edges.csv")
    print("Saved:", out_dir / "aspect_vocab.json")
    print("Saved:", out_dir / "graph_stats.json")
    if embeddings is not None:
        print("Saved:", out_dir / "review_embeddings.npy")


if __name__ == "__main__":
    main()
