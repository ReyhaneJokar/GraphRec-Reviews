
import argparse
import json
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from openai import OpenAI
from tqdm import tqdm


SYSTEM_PROMPT = """
You are an aspect-based sentiment analysis assistant for Musical Instruments reviews.

Return STRICT JSON only. No markdown, no code fences, no extra text.

For each review:
1) Extract up to {max_aspects} important aspects mentioned or implied in the review.
2) For each aspect, provide:
   - aspect: short English noun phrase (e.g., "durability", "price", "quality", "shipping")
   - sentiment: one of "positive", "neutral", "negative"
   - score: one of 1, 0, -1
   - confidence: a number between 0 and 1
   - evidence: a short phrase from the review supporting the aspect sentiment
3) Provide overall_sentiment and overall_score for the whole review.

Output format:
{{
  "overall_sentiment": "positive|neutral|negative",
  "overall_score": 1|0|-1,
  "aspects": [
    {{
      "aspect": "durability",
      "sentiment": "positive",
      "score": 1,
      "confidence": 0.93,
      "evidence": "it lasted for months"
    }}
  ]
}}
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


def extract_first_json_object(text: str) -> str:
    """
    Extract the first balanced JSON object from a string.
    Useful when the model returns extra text before/after JSON.
    """
    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if escape:
            escape = False
            continue

        if ch == "\\":
            escape = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if not in_string:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

    return text[start:]


def normalize_keys(obj):
    if isinstance(obj, dict):
        return {
            str(k).strip().strip('"').strip("'"): normalize_keys(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [normalize_keys(x) for x in obj]
    return obj


def parse_json_response(content: str) -> Dict[str, Any]:
    cleaned = clean_model_output(content)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        fallback = extract_first_json_object(content)
        data = json.loads(fallback)

    data = normalize_keys(data)
    if not isinstance(data, dict):
        raise ValueError("Model output is not a JSON object")
    return data


def normalize_absa_result(data: Dict[str, Any]) -> Dict[str, Any]:
    overall_sentiment = str(data.get("overall_sentiment", "neutral")).strip().lower()
    if overall_sentiment not in {"positive", "neutral", "negative"}:
        overall_sentiment = "neutral"

    overall_score = data.get("overall_score", 0)
    if overall_score not in {1, 0, -1}:
        if overall_sentiment == "positive":
            overall_score = 1
        elif overall_sentiment == "negative":
            overall_score = -1
        else:
            overall_score = 0

    aspects = data.get("aspects", [])
    if not isinstance(aspects, list):
        aspects = []

    clean_aspects = []
    for a in aspects:
        if not isinstance(a, dict):
            continue

        aspect = str(a.get("aspect", "")).strip()
        if not aspect:
            continue

        sentiment = str(a.get("sentiment", "neutral")).strip().lower()
        if sentiment not in {"positive", "neutral", "negative"}:
            sentiment = "neutral"

        score = a.get("score", 0)
        if score not in {1, 0, -1}:
            if sentiment == "positive":
                score = 1
            elif sentiment == "negative":
                score = -1
            else:
                score = 0

        confidence = a.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        evidence = str(a.get("evidence", "")).strip()

        clean_aspects.append(
            {
                "aspect": aspect,
                "sentiment": sentiment,
                "score": score,
                "confidence": confidence,
                "evidence": evidence,
            }
        )

    return {
        "overall_sentiment": overall_sentiment,
        "overall_score": overall_score,
        "aspects": clean_aspects,
    }


def load_existing_records(output_jsonl: Path, retry_failed: bool = False) -> Tuple[List[Dict[str, Any]], set]:
    """
    Load previously processed ABSA records from JSONL.

    Returns:
        ordered_records: records ordered by row_index
        processed_row_ids: row_index values already present
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

            if retry_failed and rec.get("error") is not None:
                continue

            by_row_index[row_index] = rec

    ordered_records = [by_row_index[k] for k in sorted(by_row_index.keys())]
    processed_row_ids = set(by_row_index.keys())
    return ordered_records, processed_row_ids


def call_llm_with_retry(
    client: OpenAI,
    model_name: str,
    review_text: str,
    max_aspects: int,
    max_tokens: int,
    temperature: float = 0.0,
    retries: int = 4,
    base_sleep: float = 1.5,
) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    prompt = SYSTEM_PROMPT.format(max_aspects=max_aspects)

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": review_text},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content or ""
            data = parse_json_response(content)
            return normalize_absa_result(data)

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


def build_record(row: pd.Series, row_index: int) -> Dict[str, Any]:
    return {
        "row_index": int(row_index),
        "user_id": safe_value(row.get("user_id", None)),
        "item_id": safe_value(row.get("item_id", None)),
        "rating": safe_value(row.get("rating", None)),
        "timestamp": safe_value(row.get("timestamp", None)),
        "review_summary": safe_value(row.get("review_summary", None)),
        "text": safe_value(row.get("text", None)),
        "absa": None,
        "error": None,
    }


def infer_row_index(row: pd.Series, idx: int) -> int:
    if "row_index" in row.index and pd.notna(row.get("row_index", None)):
        try:
            return int(row.get("row_index"))
        except Exception:
            pass
    return int(idx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Positive-only CSV produced by sentiment.py")
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--output_aspects", required=True)
    parser.add_argument("--base_url", required=True, help="OpenAI-compatible API base URL")
    parser.add_argument("--api_key", required=True, help="API key or placeholder token")
    parser.add_argument("--model", required=True, help="Model name on the API provider")
    parser.add_argument("--max_rows", type=int, default=-1, help="Limit rows for smoke test; -1 means all")
    parser.add_argument("--max_aspects", type=int, default=5)
    parser.add_argument("--max_chars", type=int, default=1500)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--checkpoint_every", type=int, default=50)
    parser.add_argument("--resume", action="store_true", help="Skip rows already present in output_jsonl")
    parser.add_argument(
        "--retry_failed",
        action="store_true",
        help="Re-run rows that previously failed and replace their old error records",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_jsonl = Path(args.output_jsonl)
    output_aspects = Path(args.output_aspects)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_aspects.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    if args.max_rows is not None and args.max_rows > 0:
        df = df.head(args.max_rows).copy()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    records: List[Dict[str, Any]] = []
    processed_row_ids = set()

    if args.resume and output_jsonl.exists():
        records, processed_row_ids = load_existing_records(output_jsonl, retry_failed=args.retry_failed)
        print(f"Resuming from {len(processed_row_ids)} already processed rows.")
        print(f"Loaded {len(records)} existing records from {output_jsonl}")

    success = 0
    failures = 0
    aspect_counter = Counter()

    for rec in records:
        if rec.get("absa") is not None:
            success += 1
            for a in rec["absa"].get("aspects", []):
                aspect = str(a.get("aspect", "")).strip().lower()
                if aspect:
                    aspect_counter[aspect] += 1
        elif rec.get("error") is not None:
            failures += 1

    mode = "a" if args.resume and output_jsonl.exists() else "w"
    with output_jsonl.open(mode, encoding="utf-8") as f:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="ABSA"):
            row_index = infer_row_index(row, idx)

            if args.resume and row_index in processed_row_ids:
                continue

            review_text = build_review_text(row, args.max_chars)
            record = build_record(row, row_index)

            try:
                absa = call_llm_with_retry(
                    client=client,
                    model_name=args.model,
                    review_text=review_text,
                    max_aspects=args.max_aspects,
                    max_tokens=args.max_tokens,
                    temperature=0.0,
                )
                record["absa"] = absa
                success += 1

                for a in absa.get("aspects", []):
                    aspect = str(a.get("aspect", "")).strip().lower()
                    if aspect:
                        aspect_counter[aspect] += 1

            except Exception as e:
                failures += 1
                record["error"] = str(e)
                print(f"\n[ERROR] row={row_index} user={record['user_id']} item={record['item_id']}")
                print(repr(e))

            records.append(record)
            processed_row_ids.add(row_index)
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

            if (row_index + 1) % args.checkpoint_every == 0:
                f.flush()

    # Rebuild outputs from de-duplicated records ordered by row_index
    if records:
        by_row_index: Dict[int, Dict[str, Any]] = {}
        for rec in records:
            try:
                rid = int(rec.get("row_index"))
            except Exception:
                continue
            by_row_index[rid] = rec

        records = [by_row_index[k] for k in sorted(by_row_index.keys())]

    # Recompute counts to make them consistent with final records
    success = 0
    failures = 0
    aspect_counter = Counter()
    for rec in records:
        if rec.get("absa") is not None:
            success += 1
            for a in rec["absa"].get("aspects", []):
                aspect = str(a.get("aspect", "")).strip().lower()
                if aspect:
                    aspect_counter[aspect] += 1
        elif rec.get("error") is not None:
            failures += 1

    with output_jsonl.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    out_df = pd.DataFrame(records)
    if not out_df.empty and "row_index" in out_df.columns:
        out_df = out_df.sort_values("row_index").drop_duplicates(subset=["row_index"], keep="last").reset_index(drop=True)

    out_df.to_csv(output_aspects.parent / f"{output_jsonl.stem}_absa.csv", index=False, encoding="utf-8-sig")

    aspects_out = {
        "total_reviews": len(df),
        "success": success,
        "failures": failures,
        "unique_aspects": len(aspect_counter),
        "aspects": [{"aspect": k, "count": v} for k, v in aspect_counter.most_common()],
        "resumed": bool(args.resume and output_jsonl.exists()),
        "processed_rows": len(processed_row_ids),
    }

    with output_aspects.open("w", encoding="utf-8") as f:
        json.dump(aspects_out, f, ensure_ascii=False, indent=2)

    print("Done.")
    print("Success:", success)
    print("Failures:", failures)
    print("Unique aspects:", len(aspect_counter))
    print("Saved:", output_jsonl)
    print("Saved:", output_aspects)


if __name__ == "__main__":
    main()
