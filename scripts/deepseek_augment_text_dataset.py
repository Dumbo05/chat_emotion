from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd

TARGET_LABELS = ("disgust", "anger", "sadness", "neutral")
EMOTIONS = ("anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral")


def normalize_space(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip())


def unique_keep_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        text = normalize_space(item)
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_json_payload(raw: str) -> object:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        starts = [pos for pos in (text.find("["), text.find("{")) if pos >= 0]
        if not starts:
            raise
        start = min(starts)
        end = max(text.rfind("]"), text.rfind("}"))
        if end <= start:
            raise
        return json.loads(text[start : end + 1])


def chat_completion(api_base: str, api_key: str, model: str, prompt: str, timeout: int) -> str:
    payload = {
        "model": model,
        "temperature": 0.7,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You generate high-quality bilingual emotion-classification paraphrases. "
                    "Return strict JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }
    request = urllib.request.Request(
        api_base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def build_prompt(batch: pd.DataFrame, variants: int) -> str:
    records = [
        {
            "id": str(row.id),
            "language": str(row.language),
            "label": str(row.label),
            "text": str(row.text),
        }
        for row in batch.itertuples(index=False)
    ]
    return (
        "Task: generate emotion-preserving paraphrases for training an emotion classifier.\n"
        f"For every input item, return exactly {variants} paraphrases.\n"
        "Rules:\n"
        "- Keep the original language: zh remains Chinese, en remains English.\n"
        "- Preserve the given label exactly: anger, disgust, sadness, or neutral.\n"
        "- Keep meaning close while changing wording naturally.\n"
        "- For neutral, do not add emotion. For anger/disgust/sadness, keep the emotion intensity.\n"
        "- Do not include the original text as a paraphrase.\n"
        "- Return only a JSON array. Each object must be {\"id\":\"...\",\"variants\":[\"...\"]}.\n\n"
        "Inputs:\n"
        + json.dumps(records, ensure_ascii=False)
    )


def normalize_response(raw: str, expected_ids: set[str], variants: int) -> dict[str, list[str]]:
    parsed = parse_json_payload(raw)
    if isinstance(parsed, dict) and "items" in parsed:
        parsed = parsed["items"]
    if not isinstance(parsed, list):
        raise ValueError("DeepSeek response is not a JSON array")
    output: dict[str, list[str]] = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        row_id = str(item.get("id", ""))
        if row_id not in expected_ids:
            continue
        values = item.get("variants", [])
        if not isinstance(values, list):
            continue
        cleaned = unique_keep_order(str(value) for value in values)[:variants]
        if cleaned:
            output[row_id] = cleaned
    return output


def load_cache(path: Path) -> dict[str, list[str]]:
    cache: dict[str, list[str]] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            cache[str(item["id"])] = unique_keep_order(item.get("variants", []))
    return cache


def append_cache(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def deterministic_fallback(text: str, label: str, language: str, variants: int) -> list[str]:
    if language == "zh":
        prefixes = {
            "anger": ["我很生气：", "这真的让人恼火：", "我对此很不满：", "太让人生气了："],
            "disgust": ["这让我觉得反感：", "我觉得很恶心：", "这件事令人厌恶：", "太让人反胃了："],
            "sadness": ["我有点难过：", "这让我很伤心：", "想到这里很失落：", "我对此感到悲伤："],
            "neutral": ["客观来说，", "只是普通陈述：", "从事实看，", "中性地说，"],
        }
        return unique_keep_order(prefix + text for prefix in prefixes[label])[:variants]
    prefixes = {
        "anger": ["I am frustrated that", "It makes me angry that", "I am annoyed because", "This irritates me because"],
        "disgust": ["It feels disgusting that", "I find it repulsive that", "It is gross that", "This feels nasty because"],
        "sadness": ["It makes me sad that", "I feel down because", "It is upsetting that", "I feel disappointed that"],
        "neutral": ["For context,", "In plain terms,", "The statement is that", "To put it neutrally,"],
    }
    lowered = text[:1].lower() + text[1:] if text else text
    return unique_keep_order(f"{prefix} {lowered}" for prefix in prefixes[label])[:variants]


def generate_cache(args: argparse.Namespace, targets: pd.DataFrame) -> dict[str, list[str]]:
    cache = load_cache(args.cache_file)
    if args.limit is not None:
        targets = targets.head(args.limit).copy()
    missing = targets[~targets["id"].astype(str).isin(cache)].copy()
    if missing.empty:
        return cache
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key env var: {args.api_key_env}")

    for start in range(0, len(missing), args.batch_size):
        batch = missing.iloc[start : start + args.batch_size]
        expected_ids = {str(value) for value in batch["id"]}
        prompt = build_prompt(batch, args.variants)
        parsed: dict[str, list[str]] = {}
        last_error = ""
        for attempt in range(1, args.max_retries + 1):
            try:
                raw = chat_completion(args.api_base, api_key, args.model, prompt, args.timeout)
                parsed = normalize_response(raw, expected_ids, args.variants)
                if expected_ids.issubset(parsed):
                    break
                last_error = "missing ids: " + ",".join(sorted(expected_ids - set(parsed)))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, ValueError) as exc:
                last_error = str(exc)
            time.sleep(max(args.sleep_seconds, min(10, attempt * 1.5)))
        if not parsed:
            print(json.dumps({"deepseek_error": last_error, "batch_start": start, "batch_size": len(batch)}, ensure_ascii=False), flush=True)
            continue
        records = [
            {"id": row_id, "variants": values[: args.variants], "model": args.model, "created_at_unix": int(time.time())}
            for row_id, values in sorted(parsed.items())
        ]
        append_cache(args.cache_file, records)
        for record in records:
            cache[record["id"]] = record["variants"]
        print(json.dumps({"deepseek_augmented": len(cache), "target_total": len(targets), "last_batch": len(records)}, ensure_ascii=False), flush=True)
        if args.sleep_seconds:
            time.sleep(args.sleep_seconds)
    return cache


def build_dataset(args: argparse.Namespace) -> dict:
    train = pd.read_csv(args.input_dir / "train.csv")
    validation = pd.read_csv(args.input_dir / "validation.csv")
    test = pd.read_csv(args.input_dir / "test.csv")
    train["text"] = train["text"].map(normalize_space)
    clean = train[(train["text"] != "") & train["label"].isin(EMOTIONS)].copy()
    clean = clean.drop_duplicates(subset=["text", "label", "language"]).copy()
    clean["source"] = "original"
    clean["parent_id"] = ""

    targets = clean[clean["label"].isin(TARGET_LABELS)].copy()
    cache = generate_cache(args, targets)

    rows: list[dict] = []
    for row in targets.itertuples(index=False):
        variants = unique_keep_order(cache.get(str(row.id), []))[: args.variants]
        source = "deepseek_paraphrase"
        if len(variants) < args.variants and args.fallback:
            variants = unique_keep_order([*variants, *deterministic_fallback(row.text, row.label, row.language, args.variants)])[: args.variants]
            source = "deepseek_paraphrase_with_template_fallback"
        for index, text in enumerate(variants, start=1):
            rows.append(
                {
                    "id": f"{row.id}-deepseek-aug-{index}",
                    "text": text,
                    "label": row.label,
                    "language": row.language,
                    "split": "train",
                    "source": source,
                    "parent_id": row.id,
                }
            )
    augmented = pd.DataFrame(rows)
    clean_train = pd.concat([clean, augmented], ignore_index=True)
    clean_train = clean_train.drop_duplicates(subset=["text", "label", "language"]).reset_index(drop=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    clean_train.to_csv(args.output_dir / "clean_train.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    clean_train[["id", "text", "label", "language", "split"]].to_csv(args.output_dir / "train.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    validation.to_csv(args.output_dir / "validation.csv", index=False)
    test.to_csv(args.output_dir / "test.csv", index=False)

    manifest = {
        "version": "dataset_v2_deepseek_aug",
        "input_dir": str(args.input_dir),
        "target_labels_augmented": list(TARGET_LABELS),
        "variants_per_sample": args.variants,
        "model": args.model,
        "api_base": args.api_base,
        "cache_file": str(args.cache_file),
        "rows": {
            "original_train": int(len(train)),
            "clean_base_train": int(len(clean)),
            "target_weak_class_rows": int(len(targets)),
            "cached_deepseek_items": int(len(cache)),
            "augmented_rows": int(len(augmented)),
            "clean_train": int(len(clean_train)),
            "validation": int(len(validation)),
            "test": int(len(test)),
        },
        "counts_by_label_clean_train": dict(Counter(clean_train["label"])),
        "counts_by_language_clean_train": dict(Counter(clean_train["language"])),
        "sha256": {
            name: sha256_file(args.output_dir / name)
            for name in ("clean_train.csv", "train.csv", "validation.csv", "test.csv")
        },
        "api_key_saved": False,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Use DeepSeek to augment weak-class bilingual text emotion data.")
    parser.add_argument("--input-dir", type=Path, default=Path("datasets/project-data/processed/dataset_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/project-data/processed/dataset_v2_deepseek_aug"))
    parser.add_argument("--cache-file", type=Path, default=Path("datasets/project-data/processed/dataset_v2_deepseek_aug/deepseek_augment_cache.jsonl"))
    parser.add_argument("--api-base", default="https://api.deepseek.com/v1")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--variants", type=int, default=4, choices=range(3, 6))
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--limit", type=int, help="Only augment first N weak-class rows; useful for smoke tests.")
    parser.add_argument("--fallback", action="store_true", help="Use local template fallback for rows not yet generated by DeepSeek.")
    args = parser.parse_args()
    manifest = build_dataset(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
