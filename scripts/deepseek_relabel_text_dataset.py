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

import pandas as pd


EMOTIONS = ("anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral")
TARGET_LABELS = ("disgust", "anger", "sadness", "neutral")


def normalize_space(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip())


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
        starts = [pos for pos in (text.find("{"), text.find("[")) if pos >= 0]
        if not starts:
            raise
        start = min(starts)
        end = max(text.rfind("}"), text.rfind("]"))
        if end <= start:
            raise
        return json.loads(text[start : end + 1])


def chat_completion(api_base: str, api_key: str, model: str, prompt: str, timeout: int) -> str:
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are cleaning emotion-classification training labels. "
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


def load_relabel_cache(path: Path) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            cache[str(item["id"])] = item
    return cache


def append_relabel_cache(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def lexicon_scores(text: str, language: str) -> dict[str, int]:
    keywords = {
        "anger": {
            "en": ("angry", "mad", "furious", "annoyed", "irritated", "hate", "rage", "ridiculous"),
            "zh": ("生气", "愤怒", "恼火", "气死", "不满", "火大"),
        },
        "disgust": {
            "en": ("disgust", "gross", "nasty", "repulsive", "revolting", "sickening", "creepy"),
            "zh": ("恶心", "厌恶", "反感", "讨厌", "反胃"),
        },
        "fear": {
            "en": ("afraid", "scared", "terrified", "fear", "panic", "anxious", "worried"),
            "zh": ("害怕", "恐惧", "惊恐", "担心", "焦虑", "吓"),
        },
        "joy": {
            "en": ("happy", "glad", "love", "great", "awesome", "wonderful", "enjoy"),
            "zh": ("开心", "高兴", "喜欢", "快乐", "愉快", "幸福"),
        },
        "sadness": {
            "en": ("sad", "unhappy", "depressed", "cry", "lonely", "miss", "disappointed"),
            "zh": ("难过", "悲伤", "伤心", "失落", "哭", "孤独", "沮丧"),
        },
        "surprise": {
            "en": ("surprise", "shocked", "wow", "unexpected", "amazed"),
            "zh": ("惊讶", "震惊", "意外", "没想到"),
        },
        "neutral": {
            "en": ("said", "says", "think", "maybe", "information", "report", "context"),
            "zh": ("认为", "表示", "说明", "信息", "情况", "事实"),
        },
    }
    lower = text.lower()
    out: dict[str, int] = {}
    for label, by_lang in keywords.items():
        out[label] = sum(1 for kw in by_lang.get(language, ()) if kw.lower() in lower)
    return out


def score_with_teacher(
    frame: pd.DataFrame,
    teacher_model_dir: Path,
    batch_size: int,
    max_length: int,
    output_path: Path,
) -> pd.DataFrame:
    if output_path.exists():
        return pd.read_csv(output_path)

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(teacher_model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(teacher_model_dir))
    id2label = getattr(model.config, "id2label", None) or {i: label for i, label in enumerate(EMOTIONS)}
    id2label = {int(k): str(v) for k, v in id2label.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    rows = frame[["id", "text", "label", "language"]].to_dict("records")

    def collate(batch: list[dict]) -> dict:
        encoded = tokenizer(
            [str(item["text"]) for item in batch],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        return {"items": batch, "encoded": encoded}

    outputs: list[dict] = []
    loader = DataLoader(rows, batch_size=batch_size, shuffle=False, collate_fn=collate)
    with torch.inference_mode():
        for batch in loader:
            encoded = {key: value.to(device) for key, value in batch["encoded"].items()}
            probs = model(**encoded).logits.softmax(dim=-1).cpu()
            conf, pred = probs.max(dim=-1)
            for item, pred_id, confidence, prob_row in zip(batch["items"], pred.tolist(), conf.tolist(), probs.tolist()):
                pred_label = id2label[int(pred_id)]
                current_id = EMOTIONS.index(str(item["label"]))
                outputs.append(
                    {
                        "id": item["id"],
                        "text": item["text"],
                        "old_label": item["label"],
                        "language": item["language"],
                        "teacher_label": pred_label,
                        "teacher_confidence": float(confidence),
                        "old_label_probability": float(prob_row[current_id]),
                    }
                )
    scored = pd.DataFrame(outputs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)
    return scored


def build_candidates(
    train: pd.DataFrame,
    scored: pd.DataFrame | None,
    max_candidates: int,
    low_confidence_threshold: float,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    duplicated_texts = train.groupby("text")["label"].nunique()
    conflict_texts = set(duplicated_texts[duplicated_texts > 1].index)

    if scored is not None:
        for row in scored.itertuples(index=False):
            reasons: list[str] = []
            if row.teacher_confidence < low_confidence_threshold:
                reasons.append("teacher_low_confidence")
            if row.teacher_label != row.old_label:
                reasons.append("teacher_disagrees")
            if row.text in conflict_texts:
                reasons.append("same_text_multiple_labels")
            if row.old_label in TARGET_LABELS or row.teacher_label in TARGET_LABELS or reasons:
                if reasons:
                    rows.append(
                        {
                            "id": row.id,
                            "text": row.text,
                            "old_label": row.old_label,
                            "language": row.language,
                            "reason": ";".join(reasons),
                            "teacher_label": row.teacher_label,
                            "teacher_confidence": row.teacher_confidence,
                            "old_label_probability": row.old_label_probability,
                        }
                    )
    else:
        for row in train.itertuples(index=False):
            scores = lexicon_scores(str(row.text), str(row.language))
            guess, score = max(scores.items(), key=lambda item: item[1])
            reasons: list[str] = []
            if row.text in conflict_texts:
                reasons.append("same_text_multiple_labels")
            if score > 0 and guess != row.label and (row.label in TARGET_LABELS or guess in TARGET_LABELS):
                reasons.append(f"lexicon_points_to_{guess}")
            if reasons:
                rows.append(
                    {
                        "id": row.id,
                        "text": row.text,
                        "old_label": row.label,
                        "language": row.language,
                        "reason": ";".join(reasons),
                        "teacher_label": guess if score > 0 else "",
                        "teacher_confidence": "",
                        "old_label_probability": "",
                    }
                )

    candidates = pd.DataFrame(rows).drop_duplicates(subset=["id"])
    if candidates.empty:
        return candidates

    priority = []
    for row in candidates.itertuples(index=False):
        score = 0
        if row.old_label in TARGET_LABELS:
            score += 3
        if "teacher_disagrees" in str(row.reason):
            score += 3
        if "teacher_low_confidence" in str(row.reason):
            score += 2
        if row.language == "zh":
            score += 1
        priority.append(score)
    candidates = candidates.assign(_priority=priority)
    candidates = candidates.sort_values(["_priority", "language", "old_label"], ascending=[False, True, True])
    if len(candidates) > max_candidates:
        candidates = candidates.head(max_candidates)
    return candidates.drop(columns=["_priority"]).sample(frac=1, random_state=seed).reset_index(drop=True)


def build_relabel_prompt(batch: pd.DataFrame) -> str:
    records = []
    for row in batch.itertuples(index=False):
        item = {
            "id": str(row.id),
            "language": str(row.language),
            "text": str(row.text),
            "current_label": str(row.old_label),
            "reason": str(row.reason),
        }
        teacher_label = str(getattr(row, "teacher_label", "") or "")
        if teacher_label:
            item["teacher_label"] = teacher_label
            item["teacher_confidence"] = getattr(row, "teacher_confidence", "")
            item["old_label_probability"] = getattr(row, "old_label_probability", "")
        records.append(item)

    return (
        "Relabel emotion-classification training samples.\n"
        f"Allowed labels: {', '.join(EMOTIONS)}.\n"
        "Rules:\n"
        "- Choose exactly one allowed label for each item.\n"
        "- If the current label is acceptable, keep it.\n"
        "- Use the text itself as the main evidence; teacher_label is only a hint.\n"
        "- For neutral, choose neutral only when no clear emotion is expressed.\n"
        "- Return strict JSON array only. Each object: "
        '{"id":"...","label":"anger","confidence":0.0-1.0,"rationale":"short"}.\n\n'
        "Items:\n"
        + json.dumps(records, ensure_ascii=False)
    )


def normalize_relabel_response(raw: str, expected_ids: set[str]) -> dict[str, dict]:
    parsed = parse_json_payload(raw)
    if isinstance(parsed, dict) and "items" in parsed:
        parsed = parsed["items"]
    if not isinstance(parsed, list):
        raise ValueError("DeepSeek response is not a JSON array")
    out: dict[str, dict] = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        row_id = str(item.get("id", ""))
        label = str(item.get("label", "")).strip().lower()
        if row_id not in expected_ids or label not in EMOTIONS:
            continue
        try:
            confidence = float(item.get("confidence", 0))
        except (TypeError, ValueError):
            confidence = 0.0
        out[row_id] = {
            "id": row_id,
            "new_label": label,
            "llm_confidence": confidence,
            "rationale": str(item.get("rationale", ""))[:300],
        }
    return out


def relabel_with_deepseek(args: argparse.Namespace, candidates: pd.DataFrame) -> dict[str, dict]:
    cache = load_relabel_cache(args.relabel_cache)
    missing = candidates[~candidates["id"].astype(str).isin(cache)].copy()
    if missing.empty:
        return cache
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key env var: {args.api_key_env}")

    for start in range(0, len(missing), args.batch_size):
        batch = missing.iloc[start : start + args.batch_size]
        expected = {str(value) for value in batch["id"]}
        prompt = build_relabel_prompt(batch)
        parsed: dict[str, dict] = {}
        last_error = ""
        for attempt in range(1, args.max_retries + 1):
            try:
                raw = chat_completion(args.api_base, api_key, args.model, prompt, args.timeout)
                parsed = normalize_relabel_response(raw, expected)
                if expected.issubset(parsed):
                    break
                last_error = "missing ids: " + ",".join(sorted(expected - set(parsed)))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, ValueError) as exc:
                last_error = str(exc)
            time.sleep(max(args.sleep_seconds, min(10, attempt * 1.5)))
        if not parsed:
            print(json.dumps({"deepseek_relabel_error": last_error, "batch_start": int(start)}, ensure_ascii=False), flush=True)
            continue
        records = []
        for row in batch.itertuples(index=False):
            item = parsed.get(str(row.id))
            if not item:
                continue
            item.update(
                {
                    "old_label": row.old_label,
                    "language": row.language,
                    "reason": row.reason,
                    "teacher_label": getattr(row, "teacher_label", ""),
                    "teacher_confidence": getattr(row, "teacher_confidence", ""),
                    "old_label_probability": getattr(row, "old_label_probability", ""),
                    "created_at_unix": int(time.time()),
                    "model": args.model,
                }
            )
            records.append(item)
        append_relabel_cache(args.relabel_cache, records)
        cache.update({record["id"]: record for record in records})
        print(json.dumps({"relabelled": len(cache), "target_total": len(candidates), "last_batch": len(records)}, ensure_ascii=False), flush=True)
        if args.sleep_seconds:
            time.sleep(args.sleep_seconds)
    return cache


def write_dataset(args: argparse.Namespace, train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, relabels: dict[str, dict]) -> dict:
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    frame = train.copy()
    frame["text"] = frame["text"].map(normalize_space)
    frame = frame[(frame["text"] != "") & frame["label"].isin(EMOTIONS)].copy()

    applied: list[dict] = []
    new_labels: list[str] = []
    for row in frame.itertuples(index=False):
        item = relabels.get(str(row.id))
        if item and item["new_label"] in EMOTIONS and float(item.get("llm_confidence", 0)) >= args.apply_confidence_threshold:
            new_label = item["new_label"]
            if new_label != row.label:
                applied.append(item)
            new_labels.append(new_label)
        else:
            new_labels.append(row.label)
    frame["label"] = new_labels
    frame = frame.drop_duplicates(subset=["text", "label", "language"]).reset_index(drop=True)

    frame.to_csv(out / "clean_train.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    frame[["id", "text", "label", "language", "split"]].to_csv(out / "train.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    validation.to_csv(out / "validation.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    pd.DataFrame(applied).to_csv(out / "applied_relabels.csv", index=False)

    manifest = {
        "version": "dataset_v3_deepseek_relabel",
        "input_dir": str(args.input_dir),
        "teacher_model_dir": str(args.teacher_model_dir) if args.teacher_model_dir else None,
        "candidate_file": str(out / "needs_deepseek_relabel.csv"),
        "relabel_cache": str(args.relabel_cache),
        "apply_confidence_threshold": args.apply_confidence_threshold,
        "rows": {
            "original_train": int(len(train)),
            "clean_train": int(len(frame)),
            "cached_relabels": int(len(relabels)),
            "applied_label_changes": int(len(applied)),
            "validation": int(len(validation)),
            "test": int(len(test)),
        },
        "counts_by_label_clean_train": dict(Counter(frame["label"])),
        "counts_by_language_clean_train": dict(Counter(frame["language"])),
        "sha256": {
            name: sha256_file(out / name)
            for name in ("clean_train.csv", "train.csv", "validation.csv", "test.csv", "applied_relabels.csv")
        },
        "api_key_saved": False,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="DeepSeek relabel suspicious text-emotion training samples.")
    parser.add_argument("--input-dir", type=Path, default=Path("datasets/project-data/processed/dataset_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/project-data/processed/dataset_v3_deepseek_relabel"))
    parser.add_argument("--teacher-model-dir", type=Path)
    parser.add_argument("--teacher-score-file", type=Path)
    parser.add_argument("--teacher-batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--low-confidence-threshold", type=float, default=0.55)
    parser.add_argument("--max-candidates", type=int, default=5000)
    parser.add_argument("--relabel-cache", type=Path, default=Path("datasets/project-data/processed/dataset_v3_deepseek_relabel/deepseek_relabel_cache.jsonl"))
    parser.add_argument("--apply-confidence-threshold", type=float, default=0.75)
    parser.add_argument("--api-base", default="https://api.deepseek.com/v1")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train = pd.read_csv(args.input_dir / "train.csv")
    validation = pd.read_csv(args.input_dir / "validation.csv")
    test = pd.read_csv(args.input_dir / "test.csv")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scored = None
    if args.teacher_model_dir:
        score_file = args.teacher_score_file or (args.output_dir / "teacher_train_scores.csv")
        scored = score_with_teacher(train, args.teacher_model_dir, args.teacher_batch_size, args.max_length, score_file)

    candidates = build_candidates(train, scored, args.max_candidates, args.low_confidence_threshold, args.seed)
    candidates.to_csv(args.output_dir / "needs_deepseek_relabel.csv", index=False)
    print(json.dumps({"candidates": int(len(candidates)), "candidate_file": str(args.output_dir / "needs_deepseek_relabel.csv")}, ensure_ascii=False), flush=True)

    relabels = relabel_with_deepseek(args, candidates)
    manifest = write_dataset(args, train, validation, test, relabels)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
