#!/usr/bin/env python
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

try:
    from groq import Groq
except ImportError as e:
    raise SystemExit(
        "Missing dependency: groq. Install it with: pip install groq"
    ) from e


JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def detect_text_column(df: pd.DataFrame) -> str:
    candidates = [
        "review_text", "text", "review", "content", "summary", "review_summary"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if "text" in c.lower() or "review" in c.lower():
            return c
    raise ValueError("No text-like column found in input CSV.")


def detect_rating_column(df: pd.DataFrame) -> str:
    candidates = ["rating:float", "rating", "score"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError("No rating column found. Expected one of rating:float / rating / score.")


def load_existing_jsonl(path: Path) -> Dict[int, dict]:
    if not path.exists():
        return {}
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "row_index" in rec:
                    out[int(rec["row_index"])] = rec
            except Exception:
                continue
    return out


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def extract_json_object(text: str) -> Optional[dict]:
    text = (text or "").strip()

    # Try direct JSON first
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try fenced / embedded JSON
    m = JSON_RE.search(text)
    if m:
        chunk = m.group(0)
        try:
            return json.loads(chunk)
        except Exception:
            return None

    return None


def build_prompt(review_text: str) -> str:
    return f"""
You are a strict sentiment classifier for product reviews.

Classify the review into one of:
- negative
- neutral
- positive

Return ONLY valid JSON with exactly these keys:
{{
  "label": "negative|neutral|positive",
  "confidence": 0.0
}}

Rules:
- confidence must be a number between 0 and 1
- output no markdown, no explanation, no extra text

Review:
\"\"\"{review_text}\"\"\"
""".strip()


def create_client(api_key: Optional[str]) -> Groq:
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit(
            "GROQ_API_KEY is not set. Set it in your environment or pass --api_key."
        )
    return Groq(api_key=api_key)


def call_groq(
    client: Groq,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout_sleep: float,
    max_retries: int,
) -> Tuple[Optional[dict], Optional[str]]:
    """
    Returns:
      (parsed_json, raw_text)
    """
    last_err = None

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a sentiment analysis assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            raw = resp.choices[0].message.content or ""
            parsed = extract_json_object(raw)
            return parsed, raw

        except Exception as e:
            last_err = e
            msg = str(e).lower()

            # rate limit / transient errors: backoff and retry
            if "429" in msg or "rate limit" in msg or "timeout" in msg or "temporarily" in msg:
                sleep_s = min(2 ** attempt, 30)
                time.sleep(sleep_s)
                continue

            # auth / quota / invalid key: stop early so you can swap key and resume
            if "401" in msg or "authentication" in msg or "unauthorized" in msg or "forbidden" in msg:
                return None, f"AUTH_ERROR: {e}"

            # other API errors: short backoff, retry
            time.sleep(timeout_sleep)

    return None, f"ERROR: {last_err}" if last_err else "ERROR"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV with reviews.")
    parser.add_argument("--output_jsonl", required=True, help="Checkpoint/output JSONL.")
    parser.add_argument("--output_csv", default=None, help="Filtered CSV of kept refined negatives.")
    parser.add_argument("--api_key", default=None, help="Groq API key (optional; GROQ_API_KEY env is recommended).")
    parser.add_argument("--model", default="llama-3.1-8b-instant")
    parser.add_argument("--negative_rating_threshold", type=float, default=1.0)
    parser.add_argument("--confidence_threshold", type=float, default=0.7)
    parser.add_argument("--max_chars", type=int, default=2500)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--max_retries", type=int, default=4)
    parser.add_argument("--flush_every", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_jsonl = Path(args.output_jsonl)
    output_csv = Path(args.output_csv) if args.output_csv else output_jsonl.with_suffix(".csv")

    df = pd.read_csv(input_path)

    text_col = detect_text_column(df)
    rating_col = detect_rating_column(df)

    if "row_index" not in df.columns:
        df = df.copy()
        df["row_index"] = df.index.astype(int)

    df["row_index"] = df["row_index"].astype(int)

    # Negative source = rating-based (baseline-compatible), but refined by Groq.
    df_neg = df[df[rating_col] <= args.negative_rating_threshold].copy()

    existing = load_existing_jsonl(output_jsonl)
    processed_row_indices = set(existing.keys())

    client = create_client(args.api_key)

    kept_records: List[dict] = []
    for rec in existing.values():
        if rec.get("kept", False):
            kept_records.append(rec)

    print(f"Input rows: {len(df)}")
    print(f"Negative candidates: {len(df_neg)}")
    print(f"Already processed (resume): {len(processed_row_indices)}")
    print(f"Using model: {args.model}")
    print(f"Output checkpoint: {output_jsonl}")
    print(f"Filtered CSV: {output_csv}")

    # Rebuild kept CSV from existing checkpoint before doing new work
    def write_filtered_csv():
        out_df = pd.DataFrame([r for r in existing.values() if r.get("kept", False)])
        if not out_df.empty:
            out_df = out_df.sort_values("row_index")
        out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    write_filtered_csv()

    processed_since_flush = 0

    try:
        for _, row in tqdm(df_neg.iterrows(), total=len(df_neg), desc="Refining negatives"):
            row_index = int(row["row_index"])
            if row_index in processed_row_indices:
                continue

            review_text = ""
            # combine summary + text if available
            if "review_summary" in row and pd.notna(row.get("review_summary", None)):
                summary = str(row.get("review_summary", "")).strip()
            else:
                summary = ""

            text = str(row.get(text_col, "")).strip()
            if summary:
                review_text = f"Summary: {summary}\nReview: {text}"
            else:
                review_text = text

            review_text = review_text[: args.max_chars]
            prompt = build_prompt(review_text)

            parsed, raw = call_groq(
                client=client,
                model=args.model,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                timeout_sleep=max(args.sleep, 0.2),
                max_retries=args.max_retries,
            )

            base_record = {
                "row_index": row_index,
                "user_id": row.get("user_id", row.get("user_id:token", None)),
                "item_id": row.get("item_id", row.get("item_id:token", None)),
                "rating": row.get(rating_col, None),
                "text_col": text_col,
                "raw_text": review_text,
                "model": args.model,
                "raw_output": raw,
                "label": None,
                "confidence": None,
                "kept": False,
                "status": "ok",
            }

            if parsed is None:
                base_record["status"] = "parse_or_api_error"
                append_jsonl(output_jsonl, base_record)
                existing[row_index] = base_record
                processed_row_indices.add(row_index)
                processed_since_flush += 1
                if processed_since_flush >= args.flush_every:
                    write_filtered_csv()
                    processed_since_flush = 0
                continue

            label = str(parsed.get("label", "")).strip().lower()
            confidence = parsed.get("confidence", 0.0)
            try:
                confidence = float(confidence)
            except Exception:
                confidence = 0.0

            if label not in {"negative", "neutral", "positive"}:
                label = "neutral"

            kept = (label == "negative") and (confidence >= args.confidence_threshold)

            base_record.update({
                "label": label,
                "confidence": confidence,
                "kept": kept,
                "status": "ok",
            })

            append_jsonl(output_jsonl, base_record)
            existing[row_index] = base_record
            processed_row_indices.add(row_index)
            processed_since_flush += 1

            if kept:
                kept_records.append(base_record)

            if processed_since_flush >= args.flush_every:
                write_filtered_csv()
                processed_since_flush = 0

            if args.sleep > 0:
                time.sleep(args.sleep)

    except KeyboardInterrupt:
        print("\nInterrupted by user. Checkpoint saved. Re-run to continue.")
    except Exception as e:
        print(f"\nStopped due to error: {e}")
        print("Checkpoint saved. Re-run with a new GROQ_API_KEY to continue from the remaining rows.")
    finally:
        # Always refresh filtered CSV from checkpoint
        write_filtered_csv()
        print(f"Saved checkpoint JSONL: {output_jsonl}")
        print(f"Saved filtered CSV: {output_csv}")
        print(f"Processed rows total: {len(existing)}")
        print(f"Kept refined negatives: {sum(1 for r in existing.values() if r.get('kept', False))}")


if __name__ == "__main__":
    main()