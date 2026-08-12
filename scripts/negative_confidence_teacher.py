import argparse
import json
import random
import re
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm


TEACHER_PROMPT = """
You are a negative-feedback confidence assistant for product reviews (Musical Instruments).

The user gave this review a rating of 1 (the lowest possible rating on a 1-5 scale),
which the recommender system currently treats as a hard "confirmed negative" signal.
Your job is to judge how confidently this review reflects genuine, durable dislike of
the product itself, versus an ambiguous case (e.g. shipping/seller complaint unrelated
to the product, a mixed review with real positives buried in it, a one-off complaint
about a single defective unit, sarcasm, or a rating that seems harsher than the text
supports).

Return STRICT JSON only. No markdown, no code fences, no extra text.

Return exactly this structure:
{
  "negative_confidence": 0.0,
  "reason": "short phrase, one sentence"
}

Rules:
- negative_confidence is a float in [0, 1]:
  1.0  = certainly a genuine, product-level negative preference
  0.5  = ambiguous / mixed signal
  0.0  = probably NOT a genuine product dislike (e.g. shipping/seller issue, isolated
         defective unit, sarcasm, or text reads neutral/positive despite the low rating)
- reason must be under 20 words and grounded in the review text
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


def normalize_confidence_result(data: Dict[str, Any]) -> Dict[str, Any]:
    conf = data.get("negative_confidence", None)
    try:
        conf = float(conf)
    except Exception:
        conf = 0.5
    conf = max(0.0, min(1.0, conf))
    reason = str(data.get("reason", "")).strip()
    return {"negative_confidence": conf, "reason": reason}


def stable_unit_score(row_index: int, seed: int) -> float:
    """
    Deterministic pseudo-random value in [0, 1) for a given row_index, stable
    across runs/machines. Used to build NESTED sample fractions: the 5% sample
    is a strict subset of the 10% sample, which is a strict subset of the 20%
    sample, etc., for the same --sample_seed. This means labeling a larger
    fraction later never re-labels rows already paid for at a smaller
    fraction -- important since the whole point of this script is to keep LLM
    calls to a minimum (idea.md's core cost argument).
    """
    h = hashlib.sha256(f"{seed}:{row_index}".encode("utf-8")).hexdigest()
    return int(h[:16], 16) / float(1 << 64)


def load_existing_records(output_jsonl: Path) -> Tuple[List[Dict[str, Any]], set]:
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
                    {"role": "system", "content": TEACHER_PROMPT},
                    {"role": "user", "content": review_text},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content or ""
            data = parse_json_response(content)
            return normalize_confidence_result(data)
        except Exception as e:
            last_error = e
            status_code = getattr(e, "status_code", None)
            response = getattr(e, "response", None)
            response_status = getattr(response, "status_code", None) if response is not None else None
            if status_code == 403 or response_status == 403:
                raise

            retry_after = None
            headers = getattr(response, "headers", None) if response is not None else None
            if headers is not None:
                retry_after = headers.get("retry-after") or headers.get("Retry-After")

            if attempt < retries - 1:
                if retry_after is not None:
                    try:
                        sleep_s = float(retry_after) + random.random()
                    except Exception:
                        sleep_s = base_sleep * (2 ** attempt) + random.random()
                else:
                    sleep_s = base_sleep * (2 ** attempt) + random.random()
                if status_code == 429 or response_status == 429:
                    print(f"  [rate limit] sleeping {sleep_s:.1f}s before retry {attempt + 2}/{retries}")
                time.sleep(sleep_s)
    raise last_error if last_error is not None else RuntimeError("Unknown LLM error")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="negative_edges.csv from prepare_graph_inputs_rating_based.py")
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--base_url", default="https://api.groq.com/openai/v1",
                         help="Groq's OpenAI-compatible endpoint. Override if needed.")
    parser.add_argument("--api_key", required=True, help="Groq API key (GROQ_API_KEY)")
    parser.add_argument("--model", default="llama-3.1-8b-instant")
    parser.add_argument("--request_delay", type=float, default=0.5,
                         help="Fixed sleep (seconds) after every call, on top of retry backoff, "
                              "to stay under Groq's per-minute rate limit.")
    parser.add_argument("--sample_frac", type=float, required=True,
                         help="Fraction of negative_edges.csv to label, e.g. 0.10 for 10%%. "
                              "Nested: 0.05 subset of 0.10 subset of 0.20, for the same --sample_seed.")
    parser.add_argument("--sample_seed", type=int, default=2024)
    parser.add_argument("--max_rows", type=int, default=-1, help="Limit rows for smoke test after sampling; -1 means all")
    parser.add_argument("--max_chars", type=int, default=1500)
    parser.add_argument("--max_tokens", type=int, default=150)
    parser.add_argument("--checkpoint_every", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_jsonl = Path(args.output_jsonl)
    output_csv = Path(args.output_csv)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if "row_index" not in df.columns:
        df = df.copy()
        df["row_index"] = df.index.astype(int)

    df["__unit_score__"] = df["row_index"].astype(int).apply(lambda r: stable_unit_score(r, args.sample_seed))
    df = df[df["__unit_score__"] < args.sample_frac].drop(columns=["__unit_score__"]).copy()
    df = df.sort_values("row_index").reset_index(drop=True)
    print(f"Sampled {len(df)} / fraction={args.sample_frac} (seed={args.sample_seed}) negative rows to label.")

    if args.max_rows is not None and args.max_rows > 0:
        df = df.head(args.max_rows).copy()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    records: List[Dict[str, Any]] = []
    processed_row_ids = set()
    if args.resume and output_jsonl.exists():
        records, processed_row_ids = load_existing_records(output_jsonl)
        print(f"Resuming from {len(processed_row_ids)} already processed rows.")

    success = 0
    failures = 0
    conf_values: List[float] = []
    for rec in records:
        if rec.get("negative_confidence") is not None:
            success += 1
            conf_values.append(rec["negative_confidence"])
        elif rec.get("error") is not None:
            failures += 1

    mode = "a" if args.resume and output_jsonl.exists() else "w"
    with output_jsonl.open(mode, encoding="utf-8") as f:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="NegConfidence"):
            row_index = int(row["row_index"])
            if args.resume and row_index in processed_row_ids:
                continue

            review_text = build_review_text(row, args.max_chars)
            record: Dict[str, Any] = {
                "row_index": row_index,
                "user_id": safe_value(row.get("user_id", None)),
                "item_id": safe_value(row.get("item_id", None)),
                "rating": safe_value(row.get("rating", None)),
                "timestamp": safe_value(row.get("timestamp", None)),
                "review_summary": safe_value(row.get("review_summary", None)),
                "text": safe_value(row.get("text", None)),
                "negative_confidence": None,
                "reason": None,
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
                conf_values.append(record["negative_confidence"])
            except Exception as e:
                failures += 1
                record["error"] = str(e)
                print(f"\n[ERROR] row={row_index} user={record['user_id']} item={record['item_id']}")
                print(repr(e))

            if args.request_delay > 0:
                time.sleep(args.request_delay)

            records.append(record)
            processed_row_ids.add(row_index)
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            if (row_index + 1) % args.checkpoint_every == 0:
                f.flush()

    out_df = pd.DataFrame(records)
    if not out_df.empty and "row_index" in out_df.columns:
        out_df = out_df.sort_values("row_index").drop_duplicates(subset=["row_index"], keep="last").reset_index(drop=True)
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    summary = {
        "sample_frac": args.sample_frac,
        "sample_seed": args.sample_seed,
        "total_sampled": len(df),
        "success": success,
        "failures": failures,
        "mean_negative_confidence": float(np.mean(conf_values)) if conf_values else None,
        "std_negative_confidence": float(np.std(conf_values)) if conf_values else None,
        "resumed": bool(args.resume and output_jsonl.exists()),
        "processed_rows": len(processed_row_ids),
    }
    summary_path = output_csv.parent / f"{output_csv.stem}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Done.")
    print("Success:", success)
    print("Failures:", failures)
    print("Mean negative_confidence:", summary["mean_negative_confidence"])
    print("Saved:", output_jsonl)
    print("Saved:", output_csv)
    print("Saved summary:", summary_path)


if __name__ == "__main__":
    main()
