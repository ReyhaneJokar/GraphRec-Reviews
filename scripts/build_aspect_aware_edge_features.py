#!/usr/bin/env python
import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def normalize_aspect(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,.;:()[]{}\"'")


def build_review_text(row: pd.Series, max_chars: int = 1500) -> str:
    summary = str(row.get("review_summary", "")).strip()
    text = str(row.get("text", "")).strip()
    if summary and summary.lower() != "nan":
        combined = f"Summary: {summary}\nReview: {text}"
    else:
        combined = text
    return combined[:max_chars]


def extract_aspects_from_absa_record(rec: dict) -> List[dict]:
    absa = rec.get("absa", None)
    if not absa:
        return []
    if isinstance(absa, str):
        absa = absa.strip()
        if not absa:
            return []
        try:
            absa = json.loads(absa)
        except Exception:
            return []
    if not isinstance(absa, dict):
        return []
    aspects = absa.get("aspects", [])
    if not isinstance(aspects, list):
        return []
    out = []
    for a in aspects:
        if not isinstance(a, dict):
            continue
        aspect = normalize_aspect(a.get("aspect", ""))
        if not aspect:
            continue
        sentiment = str(a.get("sentiment", "neutral")).strip().lower()
        if sentiment not in {"positive", "neutral", "negative"}:
            sentiment = "neutral"
        score = a.get("score", 0)
        if score not in {1, 0, -1}:
            score = 1 if sentiment == "positive" else (-1 if sentiment == "negative" else 0)
        try:
            score = int(score)
        except Exception:
            score = 0
        score = 1 if score > 0 else (-1 if score < 0 else 0)
        confidence = a.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0
        out.append({"aspect": aspect, "sentiment": sentiment, "score": score, "confidence": confidence})
    return out


def load_absa_jsonl(path: Path) -> Dict[int, List[dict]]:
    by_row: Dict[int, List[dict]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("error") is not None:
                continue
            row_index = rec.get("row_index")
            if row_index is None:
                continue
            try:
                row_index = int(row_index)
            except Exception:
                continue
            aspects = extract_aspects_from_absa_record(rec)
            if aspects:
                by_row[row_index] = aspects
    return by_row


def load_absa_csv(path: Path) -> Dict[int, List[dict]]:
    df = pd.read_csv(path)
    if "row_index" not in df.columns:
        raise ValueError(f"{path} must contain row_index column")
    by_row: Dict[int, List[dict]] = {}

    if "aspect" in df.columns:
        for _, row in df.iterrows():
            try:
                row_index = int(row["row_index"])
            except Exception:
                continue
            aspect = normalize_aspect(row.get("aspect", ""))
            if not aspect:
                continue
            sentiment = str(row.get("sentiment", "neutral")).strip().lower()
            if sentiment not in {"positive", "neutral", "negative"}:
                sentiment = "neutral"
            score = row.get("score", None)
            if pd.isna(score) or score not in {1, 0, -1, 1.0, 0.0, -1.0}:
                score = 1 if sentiment == "positive" else (-1 if sentiment == "negative" else 0)
            try:
                score = int(score)
            except Exception:
                score = 0
            score = 1 if score > 0 else (-1 if score < 0 else 0)
            conf = row.get("confidence", 0.0)
            try:
                conf = float(conf)
            except Exception:
                conf = 0.0
            by_row.setdefault(row_index, []).append({"aspect": aspect, "sentiment": sentiment, "score": score, "confidence": conf})
        return by_row

    if "absa" in df.columns:
        for _, row in df.iterrows():
            try:
                row_index = int(row["row_index"])
            except Exception:
                continue
            aspects = extract_aspects_from_absa_record(row.to_dict())
            if aspects:
                by_row[row_index] = aspects
        return by_row

    raise ValueError(f"{path} must contain either an 'aspect' column or an 'absa' column.")


def load_absa_sources(output_dir: Path, absa_jsonl: Optional[Path], absa_csv: Optional[Path]) -> Dict[int, List[dict]]:
    if absa_jsonl is not None and absa_jsonl.exists():
        return load_absa_jsonl(absa_jsonl)
    if absa_csv is not None and absa_csv.exists():
        return load_absa_csv(absa_csv)

    candidates = [
        output_dir / "positive_absa.jsonl",
        output_dir / "positive_absa_edges.csv",
        output_dir / "positive_aspect_instances.csv",
        output_dir / "absa.jsonl",
        output_dir / "absa.csv",
    ]
    for p in candidates:
        if p.exists():
            if p.suffix.lower() == ".jsonl":
                return load_absa_jsonl(p)
            return load_absa_csv(p)
    raise FileNotFoundError("No ABSA source found.")


def load_text_embeddings(
    output_dir: Path,
    sentiment_df: pd.DataFrame,
    text_embedding_file: Optional[Path],
    text_model_name: str,
    batch_size: int,
) -> Tuple[np.ndarray, Path, bool]:
    n_rows = int(sentiment_df["row_index"].max()) + 1
    candidate_paths: List[Path] = []
    if text_embedding_file is not None:
        candidate_paths.append(text_embedding_file)
    else:
        candidate_paths.extend(
            [output_dir / "review_embeddings_text_only.npy", output_dir / "review_embeddings.npy", output_dir / "text_embeddings.npy"]
        )

    for p in candidate_paths:
        if p.exists():
            emb = np.load(p)
            if emb.ndim != 2:
                raise ValueError(f"{p} must be 2D, got shape={emb.shape}")
            if emb.shape[0] < n_rows:
                raise ValueError(f"{p} has only {emb.shape[0]} rows but need {n_rows}")
            return emb[:n_rows].astype(np.float32, copy=False), p, False

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise ImportError("Install sentence-transformers or provide --text_embedding_file") from e

    model = SentenceTransformer(text_model_name)
    texts = [build_review_text(row) for _, row in sentiment_df.sort_values("row_index").iterrows()]
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)
    return emb, output_dir / f"{text_model_name.replace('/', '_')}.npy", True


def build_aspect_vocab(aspects_by_row: Dict[int, List[dict]], min_freq: int) -> Tuple[Dict[str, int], Counter]:
    counter = Counter()
    for aspects in aspects_by_row.values():
        for a in aspects:
            counter[normalize_aspect(a["aspect"])] += 1
    kept = [(asp, cnt) for asp, cnt in counter.items() if cnt >= min_freq]
    kept.sort(key=lambda x: (-x[1], x[0]))
    vocab = {asp: i for i, (asp, _) in enumerate(kept)}
    return vocab, counter


def build_aspect_matrix(
    sentiment_df: pd.DataFrame,
    aspects_by_row: Dict[int, List[dict]],
    aspect_vocab: Dict[str, int],
):
    """
    Build a signed aspect vector per review:
      +1  => aspect appears with positive sentiment
      -1  => aspect appears with negative sentiment
       0  => absent / neutralized
    """
    n_rows = int(sentiment_df["row_index"].max()) + 1
    matrix = np.zeros((n_rows, len(aspect_vocab)), dtype=np.float32)
    debug_rows = []

    for _, row in sentiment_df.sort_values("row_index").iterrows():
        row_index = int(row["row_index"])
        aspects = aspects_by_row.get(row_index, [])
        if not aspects:
            continue

        row_updates = []
        for a in aspects:
            asp = normalize_aspect(a["aspect"])
            if asp not in aspect_vocab:
                continue
            idx = aspect_vocab[asp]
            score = int(a["score"])
            score = 1 if score > 0 else (-1 if score < 0 else 0)
            if score == 0:
                continue

            prev = matrix[row_index, idx]
            if prev == 0:
                matrix[row_index, idx] = float(score)
            elif prev == score:
                pass
            else:
                # conflicting polarity for the same aspect in one review -> neutralize
                matrix[row_index, idx] = 0.0

            row_updates.append((asp, score))

        if row_updates:
            debug_rows.append(
                {
                    "row_index": row_index,
                    "num_kept_aspects": len(row_updates),
                    "kept_aspects": json.dumps(row_updates, ensure_ascii=False),
                }
            )

    return matrix, pd.DataFrame(debug_rows)


def write_json(path: Path, obj) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentiment_csv", required=True)
    parser.add_argument("--absa_jsonl", default=None)
    parser.add_argument("--absa_csv", default=None)
    parser.add_argument("--text_embedding_file", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_aspect_freq", type=int, default=10)
    parser.add_argument("--text_model", default="all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--report_only",
        action="store_true",
        help="Only report aspect frequency statistics and stop before building embeddings.",
    )
    args = parser.parse_args()

    sentiment_csv = Path(args.sentiment_csv)
    absa_jsonl = Path(args.absa_jsonl) if args.absa_jsonl else None
    absa_csv = Path(args.absa_csv) if args.absa_csv else None
    text_embedding_file = Path(args.text_embedding_file) if args.text_embedding_file else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sentiment_df = pd.read_csv(sentiment_csv)
    if "row_index" not in sentiment_df.columns:
        sentiment_df = sentiment_df.copy()
        sentiment_df["row_index"] = sentiment_df.index.astype(int)
    sentiment_df["row_index"] = sentiment_df["row_index"].astype(int)

    aspects_by_row = load_absa_sources(output_dir, absa_jsonl, absa_csv)
    aspect_vocab, aspect_counter = build_aspect_vocab(aspects_by_row, min_freq=args.min_aspect_freq)

    # Always save a lightweight report of the aspect frequencies.
    freq_report = pd.DataFrame(
        [
            {"aspect": asp, "count": int(cnt), "kept": bool(cnt >= args.min_aspect_freq)}
            for asp, cnt in sorted(aspect_counter.items(), key=lambda kv: (-kv[1], kv[0]))
        ]
    )
    freq_report.to_csv(output_dir / "aspect_frequency_report.csv", index=False, encoding="utf-8-sig")

    vocab_payload = {
        "min_aspect_freq": args.min_aspect_freq,
        "num_unique_aspects": int(len(aspect_counter)),
        "num_kept_aspects": int(len(aspect_vocab)),
        "kept_aspects": [
            {"aspect": asp, "index": idx, "count": int(aspect_counter[asp])}
            for asp, idx in sorted(aspect_vocab.items(), key=lambda kv: kv[1])
        ],
    }
    write_json(output_dir / "aspect_vocab_filtered.json", vocab_payload)

    print("Done (frequency report).")
    print(f"Unique aspects: {len(aspect_counter)}")
    print(f"Kept aspects (freq >= {args.min_aspect_freq}): {len(aspect_vocab)}")
    print(f"Saved: {output_dir / 'aspect_frequency_report.csv'}")
    print(f"Saved: {output_dir / 'aspect_vocab_filtered.json'}")

    if args.report_only:
        return

    text_embeddings, source_path, computed_flag = load_text_embeddings(
        output_dir=output_dir,
        sentiment_df=sentiment_df,
        text_embedding_file=text_embedding_file,
        text_model_name=args.text_model,
        batch_size=args.batch_size,
    )

    aspect_matrix, debug_df = build_aspect_matrix(sentiment_df, aspects_by_row, aspect_vocab)

    if text_embeddings.shape[0] != aspect_matrix.shape[0]:
        raise ValueError(
            f"Text embeddings rows ({text_embeddings.shape[0]}) and aspect matrix rows ({aspect_matrix.shape[0]}) do not match."
        )

    combined = np.concatenate(
        [text_embeddings.astype(np.float32, copy=False), aspect_matrix.astype(np.float32, copy=False)],
        axis=1,
    )

    review_embeddings_path = output_dir / "review_embeddings.npy"
    backup_path = output_dir / "review_embeddings_text_only.npy"
    if review_embeddings_path.exists() and not backup_path.exists():
        shutil.copy2(review_embeddings_path, backup_path)

    np.save(output_dir / "edge_features.npy", combined)
    np.save(review_embeddings_path, combined)
    np.save(output_dir / "aspect_vector_matrix.npy", aspect_matrix.astype(np.float32, copy=False))

    manifest = {
        "sentiment_csv": str(sentiment_csv),
        "absa_source": str(absa_jsonl if absa_jsonl else (absa_csv if absa_csv else "auto-discovered")),
        "text_embedding_source": str(source_path),
        "text_embedding_dim": int(text_embeddings.shape[1]),
        "aspect_dim": int(aspect_matrix.shape[1]),
        "combined_dim": int(combined.shape[1]),
        "min_aspect_freq": int(args.min_aspect_freq),
        "num_rows": int(combined.shape[0]),
    }
    write_json(output_dir / "edge_feature_manifest.json", manifest)

    if not debug_df.empty:
        debug_df.to_csv(output_dir / "aspect_vector_debug.csv", index=False, encoding="utf-8-sig")

    if computed_flag:
        np.save(output_dir / "review_embeddings_text_only.npy", text_embeddings.astype(np.float32, copy=False))

    print("Embedding build done.")
    print(f"Text dim: {text_embeddings.shape[1]}")
    print(f"Aspect dim: {aspect_matrix.shape[1]}")
    print(f"Combined dim: {combined.shape[1]}")
    print(f"Saved: {review_embeddings_path}")
    print(f"Saved: {output_dir / 'edge_features.npy'}")
    print(f"Saved: {output_dir / 'aspect_vector_matrix.npy'}")
    print(f"Saved: {output_dir / 'edge_feature_manifest.json'}")


if __name__ == "__main__":
    main()
