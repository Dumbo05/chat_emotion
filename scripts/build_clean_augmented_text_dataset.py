from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd


EMOTIONS = ("anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral")
TARGET_LABELS = ("disgust", "anger", "sadness", "neutral")


EN_REPLACEMENTS: dict[str, list[tuple[str, str]]] = {
    "anger": [
        (r"\bangry\b", "mad"),
        (r"\bmad\b", "angry"),
        (r"\bannoying\b", "irritating"),
        (r"\birritating\b", "annoying"),
        (r"\bhate\b", "cannot stand"),
        (r"\bridiculous\b", "absurd"),
        (r"\bawful\b", "infuriating"),
    ],
    "disgust": [
        (r"\bgross\b", "disgusting"),
        (r"\bdisgusting\b", "gross"),
        (r"\bnasty\b", "repulsive"),
        (r"\brevolting\b", "disgusting"),
        (r"\bsick\b", "nauseating"),
        (r"\bcreepy\b", "repulsive"),
    ],
    "sadness": [
        (r"\bsad\b", "unhappy"),
        (r"\bunhappy\b", "sad"),
        (r"\bupset\b", "heartbroken"),
        (r"\bdisappointed\b", "let down"),
        (r"\blonely\b", "alone"),
        (r"\bmiss\b", "really miss"),
    ],
    "neutral": [
        (r"\bI think\b", "I believe"),
        (r"\bmaybe\b", "perhaps"),
        (r"\bokay\b", "fine"),
        (r"\bsaid\b", "mentioned"),
        (r"\buse\b", "utilize"),
        (r"\bshow\b", "display"),
    ],
}


EN_PREFIXES = {
    "anger": ["I am frustrated that", "It makes me angry that", "I am annoyed because"],
    "disgust": ["It feels disgusting that", "I find it repulsive that", "It is gross that"],
    "sadness": ["It makes me sad that", "I feel down because", "It is upsetting that"],
    "neutral": ["For context,", "In plain terms,", "The statement is that"],
}


ZH_PREFIXES = {
    "anger": ["我很生气：", "这真的让人恼火：", "我对此很不满：", "太让人生气了："],
    "disgust": ["这让我觉得反感：", "我觉得很恶心：", "这件事令人厌恶：", "太让人反胃了："],
    "sadness": ["我有点难过：", "这让我很伤心：", "想到这里很失落：", "我对此感到悲伤："],
    "neutral": ["客观来说，", "只是普通陈述：", "从事实看，", "中性地说，"],
}


ZH_SUFFIXES = {
    "anger": ["，这真的让人恼火。", "，我很不满。", "，太让人生气了。"],
    "disgust": ["，这让我很反感。", "，真让人恶心。", "，令人厌恶。"],
    "sadness": ["，这让我很难过。", "，我感到失落。", "，听起来很伤心。"],
    "neutral": ["，这是一个中性表达。", "，没有明显情绪。", "，只是陈述事实。"],
}


LEXICON = {
    "anger": {
        "en": ("angry", "mad", "furious", "annoyed", "irritated", "hate", "rage", "ridiculous"),
        "zh": ("生气", "愤怒", "恼火", "气死", "不满", "火大"),
    },
    "disgust": {
        "en": ("disgust", "gross", "nasty", "repulsive", "revolting", "sickening", "creepy"),
        "zh": ("恶心", "厌恶", "反感", "恶臭", "讨厌", "反胃"),
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_space(text: object) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def apply_replacements(text: str, label: str, limit: int) -> list[str]:
    variants: list[str] = []
    for pattern, repl in EN_REPLACEMENTS.get(label, []):
        candidate = re.sub(pattern, repl, text, count=1, flags=re.IGNORECASE)
        if candidate != text:
            variants.append(candidate)
        if len(variants) >= limit:
            break
    return variants


def sentence_case(text: str) -> str:
    if not text:
        return text
    return text[0].upper() + text[1:]


def paraphrase_en(text: str, label: str, count: int) -> list[str]:
    variants = apply_replacements(text, label, count)
    prefixes = EN_PREFIXES[label]
    for prefix in prefixes:
        if len(variants) >= count:
            break
        core = text
        if core and core[0].isupper():
            core = core[0].lower() + core[1:]
        variants.append(f"{prefix} {core}")
    if len(variants) < count:
        variants.append(sentence_case(text.replace("!", ".")))
    if len(variants) < count:
        variants.append(f"{text} This is how I feel about it.")
    return unique_keep_order(variants)[:count]


def paraphrase_zh(text: str, label: str, count: int) -> list[str]:
    variants: list[str] = []
    for prefix in ZH_PREFIXES[label]:
        variants.append(prefix + text)
        if len(variants) >= count:
            break
    for suffix in ZH_SUFFIXES[label]:
        if len(variants) >= count:
            break
        variants.append(text.rstrip("。！？!?") + suffix)
    return unique_keep_order(variants)[:count]


def unique_keep_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        item = normalize_space(item)
        if item and item not in seen:
            seen.add(item)
            output.append(item)
    return output


def lexicon_prediction(text: str, language: str) -> tuple[str | None, dict[str, int]]:
    lowered = text.lower()
    scores: dict[str, int] = {}
    for label, by_language in LEXICON.items():
        score = 0
        for keyword in by_language.get(language, ()):
            if keyword.lower() in lowered:
                score += 1
        scores[label] = score
    best_label, best_score = max(scores.items(), key=lambda item: item[1])
    return (best_label if best_score > 0 else None), scores


def find_relabel_candidates(frame: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    rows: list[dict] = []
    duplicated_texts = frame.groupby("text")["label"].nunique()
    conflict_texts = set(duplicated_texts[duplicated_texts > 1].index)
    for row in frame.itertuples(index=False):
        guessed, scores = lexicon_prediction(row.text, row.language)
        reasons: list[str] = []
        if row.text in conflict_texts:
            reasons.append("same_text_multiple_labels")
        if guessed is not None and guessed != row.label:
            if row.label in TARGET_LABELS or guessed in TARGET_LABELS:
                reasons.append(f"lexicon_points_to_{guessed}")
        if len(str(row.text).split()) <= 2 and row.language == "en" and row.label != "neutral":
            reasons.append("very_short_non_neutral_en")
        if reasons:
            rows.append(
                {
                    "id": row.id,
                    "text": row.text,
                    "old_label": row.label,
                    "language": row.language,
                    "reason": ";".join(reasons),
                    "lexicon_guess": guessed or "",
                    "lexicon_scores": json.dumps(scores, ensure_ascii=False),
                }
            )
    candidates = pd.DataFrame(rows)
    if len(candidates) > max_rows:
        candidates = candidates.sample(n=max_rows, random_state=seed)
    return candidates.sort_values(["language", "old_label", "id"]).reset_index(drop=True)


def read_manual_relabels(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    frame = pd.read_csv(path)
    required = {"id", "new_label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manual relabel file is missing columns: {sorted(missing)}")
    mapping: dict[str, str] = {}
    for row in frame.itertuples(index=False):
        new_label = str(row.new_label).strip()
        if new_label and new_label in EMOTIONS:
            mapping[str(row.id)] = new_label
    return mapping


def chat_completion(api_base: str, api_key: str, model: str, prompt: str) -> str:
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You relabel emotion classification training data. "
                    "Return strict JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }
    request = urllib.request.Request(
        api_base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def relabel_with_llm(
    candidates: pd.DataFrame,
    api_base: str | None,
    api_key_env: str,
    model: str,
    sleep_seconds: float,
) -> dict[str, str]:
    if not api_base:
        return {}
    api_key = os.environ.get(api_key_env)
    if not api_key:
        return {}
    output: dict[str, str] = {}
    labels = ", ".join(EMOTIONS)
    for row in candidates.itertuples(index=False):
        prompt = (
            f"Allowed labels: {labels}\n"
            f"Language: {row.language}\n"
            f"Current label: {row.old_label}\n"
            f"Text: {row.text}\n\n"
            "Choose the single best emotion label. If the current label is acceptable, keep it. "
            'Return JSON like {"label":"anger","confidence":0.92,"rationale":"short"}'
        )
        try:
            raw = chat_completion(api_base, api_key, model, prompt)
            parsed = json.loads(raw)
            label = str(parsed.get("label", "")).strip().lower()
            confidence = float(parsed.get("confidence", 0))
            if label in EMOTIONS and confidence >= 0.7:
                output[str(row.id)] = label
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, ValueError) as exc:
            print(json.dumps({"llm_relabel_error": str(exc), "id": row.id}, ensure_ascii=False), flush=True)
        if sleep_seconds:
            time.sleep(sleep_seconds)
    return output


def make_augmented_rows(frame: pd.DataFrame, per_sample: int, seed: int) -> pd.DataFrame:
    random.seed(seed)
    rows: list[dict] = []
    for row in frame.itertuples(index=False):
        if row.label not in TARGET_LABELS:
            continue
        variants = (
            paraphrase_zh(row.text, row.label, per_sample)
            if row.language == "zh"
            else paraphrase_en(row.text, row.label, per_sample)
        )
        for index, text in enumerate(variants, start=1):
            rows.append(
                {
                    "id": f"{row.id}-aug-{index}",
                    "text": text,
                    "label": row.label,
                    "language": row.language,
                    "split": "train",
                    "source": "weak_class_paraphrase",
                    "parent_id": row.id,
                }
            )
    return pd.DataFrame(rows)


def clean_base_train(train: pd.DataFrame, relabels: dict[str, str]) -> pd.DataFrame:
    frame = train.copy()
    frame["text"] = frame["text"].map(normalize_space)
    frame = frame[frame["text"] != ""].copy()
    frame = frame[frame["label"].isin(EMOTIONS)].copy()
    frame["label"] = [relabels.get(str(row_id), label) for row_id, label in zip(frame["id"], frame["label"])]
    frame = frame.drop_duplicates(subset=["text", "label", "language"]).copy()
    frame["source"] = "original_or_relabelled"
    frame["parent_id"] = ""
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Build clean_train.csv with weak-class bilingual augmentation.")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/project-data/processed/dataset_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/project-data/processed/dataset_v2_clean_aug"))
    parser.add_argument("--aug-per-sample", type=int, default=4, choices=range(3, 6))
    parser.add_argument("--max-relabel-candidates", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manual-relabels", type=Path)
    parser.add_argument("--llm-api-base", help="OpenAI-compatible API base, for example https://api.openai.com/v1")
    parser.add_argument("--llm-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--llm-model", default="gpt-4.1-mini")
    parser.add_argument("--llm-sleep-seconds", type=float, default=0.0)
    args = parser.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv")
    validation = pd.read_csv(args.data_dir / "validation.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates = find_relabel_candidates(train, args.max_relabel_candidates, args.seed)
    candidates.to_csv(args.output_dir / "needs_llm_relabel.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    relabels = read_manual_relabels(args.manual_relabels)
    llm_relabels = relabel_with_llm(
        candidates,
        api_base=args.llm_api_base,
        api_key_env=args.llm_api_key_env,
        model=args.llm_model,
        sleep_seconds=args.llm_sleep_seconds,
    )
    relabels.update(llm_relabels)

    clean = clean_base_train(train, relabels)
    augmented = make_augmented_rows(clean, args.aug_per_sample, args.seed)
    clean_train = pd.concat([clean, augmented], ignore_index=True)
    clean_train = clean_train.drop_duplicates(subset=["text", "label", "language"]).reset_index(drop=True)

    clean_train.to_csv(args.output_dir / "clean_train.csv", index=False)
    clean_train[["id", "text", "label", "language", "split"]].to_csv(args.output_dir / "train.csv", index=False)
    validation.to_csv(args.output_dir / "validation.csv", index=False)
    test.to_csv(args.output_dir / "test.csv", index=False)

    relabel_frame = pd.DataFrame(
        [{"id": key, "new_label": value, "source": "manual_or_llm"} for key, value in sorted(relabels.items())]
    )
    relabel_frame.to_csv(args.output_dir / "applied_relabels.csv", index=False)

    manifest = {
        "version": "dataset_v2_clean_aug",
        "source_data_dir": str(args.data_dir),
        "target_labels_augmented": list(TARGET_LABELS),
        "augmentation_per_sample": args.aug_per_sample,
        "llm_relabel_api_used": bool(args.llm_api_base and os.environ.get(args.llm_api_key_env)),
        "manual_relabel_file": str(args.manual_relabels) if args.manual_relabels else None,
        "rows": {
            "original_train": int(len(train)),
            "clean_base_train": int(len(clean)),
            "augmented_rows": int(len(augmented)),
            "clean_train": int(len(clean_train)),
            "validation": int(len(validation)),
            "test": int(len(test)),
            "relabel_candidates": int(len(candidates)),
            "applied_relabels": int(len(relabels)),
        },
        "counts_by_label_clean_train": dict(Counter(clean_train["label"])),
        "counts_by_language_clean_train": dict(Counter(clean_train["language"])),
        "sha256": {
            name: sha256_file(args.output_dir / name)
            for name in ("clean_train.csv", "train.csv", "validation.csv", "test.csv", "needs_llm_relabel.csv")
        },
        "note": (
            "If llm_relabel_api_used is false, needs_llm_relabel.csv is the queue for external LLM review; "
            "no unsupported relabels were fabricated."
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
