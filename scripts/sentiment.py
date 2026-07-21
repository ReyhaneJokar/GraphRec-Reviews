import argparse
import json
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd
from openai import OpenAI
from tqdm import tqdm


SENTIMENT_PROMPT = """
You are a sentiment analysis assistant for product reviews.

Return STRICT JSON only. No markdown, no code fences, no extra text.

Classify the overall sentiment of the review as one of:
- positive
- neutral
- negative

Return exactly this structure:
{
  "sentiment": "positive|neutral|negative",
  "score": 1|0|-1,
  "confidence": 0.0,
  "evidence": "short phrase from the review"
}

Rules:
- sentiment must be one of positive, neutral, negative
- score must match sentiment: positive=1, neutral=0, negative=-1
- confidence must be a number between 0 and 1
- evidence should be a short phrase copied or closely paraphrased from the review
"""


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


def build_review_text(row: pd.Series, max_chars: int) -> str:
    summary = str(row.get("review_summary", "")).strip()
    text = str(row.get("text", "")).strip()

    if summary and summary.lower() != "nan":
        combined = f"Summary: {summary}\nReview: {text}"
    else:
        combined = text

    return combined[:max_chars]


def clean_model_output(content: str) -> str:
    content = content.strip()

    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"\s*```$", "", content).strip()

    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        return content[start:end + 1].strip()

    return content


def parse_json_response(content: str) -> Dict[str, Any]:
    cleaned = clean_model_output(content)
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("Model output is not a JSON object")
    return data


def normalize_sentiment_result(data: Dict[str, Any]) -> Dict[str, Any]:
    sentiment = str(data.get("sentiment", "neutral")).strip().lower()
    if sentiment not in {"positive", "neutral", "negative"}:
        sentiment = "neutral"

    score = data.get("score", None)
    if score not in {1, 0, -1}:
        if sentiment == "positive":
            score = 1
        elif sentiment == "negative":
            score = -1
        else:
            score = 0

    confidence = data.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    evidence = str(data.get("evidence", "")).strip()

    return {
        "sentiment": sentiment,
        "score": score,
        "confidence": confidence,
        "evidence": evidence,
    }


def load_existing_records(output_jsonl: Path) -> Tuple[List[Dict[str, Any]], set]:
    """
    Read previously processed records from a JSONL file.

    Returns:
        ordered_records: records ordered by row_index
        processed_row_ids: set of row_index values already present in file
    """
    by_row_index: Dict[int, Dict[str, Any]] = {}

    if not output_jsonl.exists():
        return [], set()

    with output_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            row_index = rec.get("row_index", None)
            if row_index is None:
                continue
            try:
                row_index = int(row_index)
            except Exception:
                continue
            rec["row_index"] = row_index
            by_row_index[row_index] = rec

    ordered_records = [by_row_index[k] for k in sorted(by_row_index.keys())]
    processed_row_ids = set(by_row_index.keys())
    return ordered_records, processed_row_ids


def call_llm_with_retry(
    client: OpenAI,
    model_name: str,
    review_text: str,
    max_tokens: int,
    temperature: float = 0.0,
    retries: int = 4,
    base_sleep: float = 1.5,
) -> Dict[str, Any]:
    last_error: Optional[Exception] = None

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SENTIMENT_PROMPT},
                    {"role": "user", "content": review_text},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content or ""
            data = parse_json_response(content)
            return normalize_sentiment_result(data)

        except Exception as e:
            last_error = e

            status_code = getattr(e, "status_code", None)
            response = getattr(e, "response", None)
            response_status = getattr(response, "status_code", None) if response is not None else None
            if status_code == 403 or response_status == 403:
                raise

            if attempt < retries - 1:
                sleep_s = base_sleep * (2 ** attempt) + random.random()
                time.sleep(sleep_s)

    raise last_error if last_error is not None else RuntimeError("Unknown LLM error")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--base_url", required=True)
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max_rows", type=int, default=-1, help="Limit rows for smoke test; -1 means all")
    parser.add_argument("--max_chars", type=int, default=1500)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--checkpoint_every", type=int, default=50)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output_jsonl and continue from the next row_index",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_jsonl = Path(args.output_jsonl)
    output_csv = Path(args.output_csv)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    if args.max_rows is not None and args.max_rows > 0:
        df = df.head(args.max_rows).copy()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    records: List[Dict[str, Any]] = []
    processed_row_ids = set()

    if args.resume and output_jsonl.exists():
        records, processed_row_ids = load_existing_records(output_jsonl)
        print(f"Resuming from {len(processed_row_ids)} already processed rows.")
        print(f"Loaded {len(records)} existing records from {output_jsonl}")

    success = 0
    failures = 0
    counter = Counter()

    for rec in records:
        if rec.get("sentiment") in {"positive", "negative", "neutral"}:
            success += 1
            counter[rec["sentiment"]] += 1
        elif rec.get("error") is not None:
            failures += 1

    mode = "a" if args.resume and output_jsonl.exists() else "w"
    with output_jsonl.open(mode, encoding="utf-8") as f:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Sentiment"):
            idx_int = int(idx)
            if args.resume and idx_int in processed_row_ids:
                continue

            review_text = build_review_text(row, args.max_chars)

            record: Dict[str, Any] = {
                "row_index": idx_int,
                "user_id": safe_value(row.get("user_id", None)),
                "item_id": safe_value(row.get("item_id", None)),
                "rating": safe_value(row.get("rating", None)),
                "timestamp": safe_value(row.get("timestamp", None)),
                "review_summary": safe_value(row.get("review_summary", None)),
                "text": safe_value(row.get("text", None)),
                "sentiment": None,
                "score": None,
                "confidence": None,
                "evidence": None,
                "error": None,
            }

            try:
                result = call_llm_with_retry(
                    client=client,
                    model_name=args.model,
                    review_text=review_text,
                    max_tokens=args.max_tokens,
                    temperature=0.0,
                )
                record.update(result)
                success += 1
                counter[record["sentiment"]] += 1

            except Exception as e:
                failures += 1
                record["error"] = str(e)
                print(f"\n[ERROR] row={idx_int} user={record['user_id']} item={record['item_id']}")
                print(repr(e))

            records.append(record)
            processed_row_ids.add(idx_int)
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

            if (idx_int + 1) % args.checkpoint_every == 0:
                f.flush()

    out_df = pd.DataFrame(records)
    if not out_df.empty and "row_index" in out_df.columns:
        out_df = out_df.sort_values("row_index").drop_duplicates(subset=["row_index"], keep="last").reset_index(drop=True)

    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    stem = output_csv.with_suffix("")
    for label in ["positive", "negative", "neutral"]:
        subset = out_df[out_df["sentiment"] == label].copy()
        subset_path = Path(f"{stem}_{label}.csv")
        subset.to_csv(subset_path, index=False, encoding="utf-8-sig")

    summary = {
        "total_reviews": len(df),
        "success": success,
        "failures": failures,
        "counts": dict(counter),
        "resumed": bool(args.resume and output_jsonl.exists()),
        "processed_rows": len(processed_row_ids),
    }
    summary_path = output_csv.parent / f"{output_csv.stem}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Done.")
    print("Success:", success)
    print("Failures:", failures)
    print("Counts:", dict(counter))
    print("Saved:", output_jsonl)
    print("Saved:", output_csv)
    print("Saved positive/negative/neutral splits next to output_csv")
    print("Saved summary:", summary_path)


if __name__ == "__main__":
    main()
