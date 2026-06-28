from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


EMOTIONS = ("anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral")

LABEL_MAP = {
    "angry": "anger",
    "anger": "anger",
    "愤怒": "anger",
    "生气": "anger",
    "disgust": "disgust",
    "disgusted": "disgust",
    "厌恶": "disgust",
    "反感": "disgust",
    "fear": "fear",
    "fearful": "fear",
    "scared": "fear",
    "恐惧": "fear",
    "害怕": "fear",
    "happy": "joy",
    "happiness": "joy",
    "joy": "joy",
    "喜悦": "joy",
    "高兴": "joy",
    "开心": "joy",
    "sad": "sadness",
    "sadness": "sadness",
    "悲伤": "sadness",
    "难过": "sadness",
    "surprise": "surprise",
    "surprised": "surprise",
    "惊讶": "surprise",
    "neutral": "neutral",
    "none": "neutral",
    "other": "neutral",
    "中性": "neutral",
    "无情绪": "neutral",
}

TEXT_COLUMNS = ("text", "utterance", "sentence", "content", "transcript", "query", "dialogue", "文本", "内容")
LABEL_COLUMNS = ("label", "emotion", "Emotion", "sentiment", "category", "情绪", "标签")
LANG_COLUMNS = ("language", "lang")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip())


def map_label(value: Any) -> str | None:
    raw = normalize_space(value).strip("'\"").lower()
    return LABEL_MAP.get(raw)


def infer_language(text: str, default: str | None = None) -> str:
    if default in {"en", "zh"}:
        return default
    chinese_chars = sum("\u4e00" <= ch <= "\u9fff" for ch in text)
    return "zh" if chinese_chars > 0 else "en"


def flatten_json(value: Any) -> list[dict]:
    rows: list[dict] = []
    if isinstance(value, dict):
        if any(key in value for key in TEXT_COLUMNS) and any(key in value for key in LABEL_COLUMNS):
            rows.append(value)
        for item in value.values():
            rows.extend(flatten_json(item))
    elif isinstance(value, list):
        for item in value:
            rows.extend(flatten_json(item))
    return rows


def read_any(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(path, sep=sep, engine="python", on_bad_lines="skip")
    if suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return pd.DataFrame(rows)
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return pd.DataFrame(flatten_json(data))
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported supplement file type: {path}")


def first_existing(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lower_to_original = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    return None


def normalize_supplement(path: Path, default_language: str | None, source_name: str) -> pd.DataFrame:
    raw = read_any(path)
    text_col = first_existing(list(raw.columns), TEXT_COLUMNS)
    label_col = first_existing(list(raw.columns), LABEL_COLUMNS)
    lang_col = first_existing(list(raw.columns), LANG_COLUMNS)
    if text_col is None or label_col is None:
        raise ValueError(f"{path} must contain a text column and a label/emotion column.")

    rows: list[dict] = []
    for idx, row in raw.iterrows():
        text = normalize_space(row[text_col])
        label = map_label(row[label_col])
        if not text or label not in EMOTIONS:
            continue
        language = infer_language(text, normalize_space(row[lang_col]).lower() if lang_col else default_language)
        rows.append(
            {
                "id": f"supp-{source_name}-{idx}",
                "text": text,
                "label": label,
                "language": language,
                "split": "train",
                "source": source_name,
            }
        )
    return pd.DataFrame(rows)


def parse_cap(value: str) -> dict[tuple[str, str], int]:
    caps: dict[tuple[str, str], int] = {}
    if not value:
        return caps
    for part in value.split(","):
        key, raw_limit = part.split("=")
        label, language = key.split("/")
        caps[(label.strip(), language.strip())] = int(raw_limit)
    return caps


def apply_caps(frame: pd.DataFrame, caps: dict[tuple[str, str], int]) -> pd.DataFrame:
    if not caps or frame.empty:
        return frame
    used: defaultdict[tuple[str, str], int] = defaultdict(int)
    keep: list[int] = []
    for idx, row in frame.iterrows():
        key = (row["label"], row["language"])
        limit = caps.get(key, caps.get((row["label"], "*"), caps.get(("*", row["language"]), caps.get(("*", "*"), 10**9))))
        if used[key] < limit:
            used[key] += 1
            keep.append(idx)
    return frame.loc[keep].reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge supplemental emotion datasets into dataset_v1 train split.")
    parser.add_argument("--base-dir", type=Path, default=Path("datasets/project-data/processed/dataset_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/project-data/processed/dataset_v5_supplemented"))
    parser.add_argument("--supplement", action="append", default=[], help="Format: path[:language[:source_name]]")
    parser.add_argument(
        "--caps",
        default="neutral/zh=5000,neutral/en=3000,anger/zh=2500,disgust/zh=2500,fear/zh=2500,surprise/zh=2500,anger/en=1500,disgust/en=1500,fear/en=1500,surprise/en=1500,*/zh=4000,*/en=2500",
        help="Comma separated caps like neutral/zh=5000,*/en=2500.",
    )
    args = parser.parse_args()

    train = pd.read_csv(args.base_dir / "train.csv")
    validation = pd.read_csv(args.base_dir / "validation.csv")
    test = pd.read_csv(args.base_dir / "test.csv")

    supplements: list[pd.DataFrame] = []
    for item in args.supplement:
        parts = item.split(":")
        path = Path(parts[0])
        default_language = parts[1] if len(parts) >= 2 and parts[1] else None
        source_name = parts[2] if len(parts) >= 3 and parts[2] else path.stem
        supplements.append(normalize_supplement(path, default_language, source_name))

    supplement_frame = pd.concat(supplements, ignore_index=True) if supplements else pd.DataFrame(
        columns=["id", "text", "label", "language", "split", "source"]
    )
    supplement_frame = supplement_frame.drop_duplicates(subset=["text", "label", "language"]).reset_index(drop=True)
    supplement_frame = apply_caps(supplement_frame, parse_cap(args.caps))

    train2 = train.copy()
    train2["source"] = "dataset_v1"
    merged = pd.concat([train2, supplement_frame], ignore_index=True)
    merged = merged.drop_duplicates(subset=["text", "label", "language"]).reset_index(drop=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged[["id", "text", "label", "language", "split"]].to_csv(args.output_dir / "train.csv", index=False)
    merged.to_csv(args.output_dir / "clean_train.csv", index=False)
    validation.to_csv(args.output_dir / "validation.csv", index=False)
    test.to_csv(args.output_dir / "test.csv", index=False)
    supplement_frame.to_csv(args.output_dir / "accepted_supplements.csv", index=False)

    manifest = {
        "version": "dataset_v5_supplemented",
        "base_dir": str(args.base_dir),
        "supplements": args.supplement,
        "caps": args.caps,
        "rows": {
            "base_train": int(len(train)),
            "accepted_supplements": int(len(supplement_frame)),
            "merged_train": int(len(merged)),
            "validation": int(len(validation)),
            "test": int(len(test)),
        },
        "supplement_counts_by_label_language": {
            f"{label}/{language}": int(count)
            for (label, language), count in Counter(zip(supplement_frame["label"], supplement_frame["language"])).items()
        },
        "merged_counts_by_label_language": {
            f"{label}/{language}": int(count)
            for (label, language), count in Counter(zip(merged["label"], merged["language"])).items()
        },
        "sha256": {
            name: sha256_file(args.output_dir / name)
            for name in ("train.csv", "clean_train.csv", "validation.csv", "test.csv", "accepted_supplements.csv")
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
