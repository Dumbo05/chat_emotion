from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emotion_app.domain import EMOTIONS, normalize_emotion


CHINESE_LABELS = {
    "愤怒": "anger",
    "生气": "anger",
    "厌恶": "disgust",
    "恐惧": "fear",
    "害怕": "fear",
    "喜悦": "joy",
    "愉快": "joy",
    "开心": "joy",
    "高兴": "joy",
    "喜欢": "joy",
    "悲伤": "sadness",
    "难过": "sadness",
    "惊讶": "surprise",
    "中性": "neutral",
    "无情绪": "neutral",
    "其他": "neutral",
    "like": "joy",
    "love": "joy",
}


def map_label(value: object) -> str:
    raw = str(value).strip()
    if raw in CHINESE_LABELS:
        return CHINESE_LABELS[raw]
    return normalize_emotion(raw)


def read_goemotions(directory: Path) -> pd.DataFrame:
    labels = [line.strip() for line in (directory / "labels.txt").read_text(encoding="utf-8").splitlines()]
    frames: list[pd.DataFrame] = []
    for filename, split in (("train.tsv", "train"), ("dev.tsv", "validation"), ("test.tsv", "test")):
        source = pd.read_csv(
            directory / filename,
            sep="\t",
            header=None,
            names=["text", "label_ids", "source_id"],
            dtype=str,
        )
        # This application is single-label. Multi-label rows are deliberately
        # excluded rather than collapsed or assigned an arbitrary first label.
        source = source[~source["label_ids"].str.contains(",", regex=False)].copy()
        source["label"] = source["label_ids"].map(lambda value: map_label(labels[int(value)]))
        source["id"] = source["source_id"].map(lambda value: f"en-goemotions-{value}")
        source["language"] = "en"
        source["split"] = split
        frames.append(source[["id", "text", "label", "language", "split"]])
    return pd.concat(frames, ignore_index=True)


def read_chinese(path: Path, seed: int) -> pd.DataFrame:
    source = pd.read_csv(path)
    text_column = next((name for name in ("text", "content", "文本", "内容") if name in source.columns), None)
    label_column = next((name for name in ("label", "emotion", "情绪", "标签") if name in source.columns), None)
    if text_column is None or label_column is None:
        raise ValueError("中文 CSV 必须包含文本列和标签列。")
    frame = pd.DataFrame()
    frame["text"] = source[text_column].fillna("").astype(str).str.strip()
    frame["label"] = source[label_column].map(map_label)
    if "id" in source.columns:
        frame["id"] = source["id"].astype(str).map(lambda value: f"zh-{value}")
    else:
        frame["id"] = [f"zh-{index:08d}" for index in range(len(frame))]
    frame["language"] = "zh"
    if "split" in source.columns:
        aliases = {"dev": "validation", "val": "validation"}
        frame["split"] = source["split"].astype(str).str.lower().replace(aliases)
    frame = frame[frame["text"] != ""].drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    if "split" in frame.columns:
        if not set(frame["split"]).issubset({"train", "validation", "test"}):
            raise ValueError("中文 CSV 的 split 只能是 train、validation/dev/val 或 test。")
    else:
        train, remainder = train_test_split(
            frame, test_size=0.2, random_state=seed, stratify=frame["label"]
        )
        validation, test = train_test_split(
            remainder, test_size=0.5, random_state=seed, stratify=remainder["label"]
        )
        train = train.assign(split="train")
        validation = validation.assign(split="validation")
        test = test.assign(split="test")
        frame = pd.concat([train, validation, test], ignore_index=True)
    return frame[["id", "text", "label", "language", "split"]]


def write_outputs(frame: pd.DataFrame, output_dir: Path, sources: list[str], seed: int) -> None:
    frame = frame.drop_duplicates(subset=["id"]).copy()
    if not set(frame["label"]).issubset(EMOTIONS):
        raise ValueError("数据中包含非标准标签。")
    if set(frame["split"]) != {"train", "validation", "test"}:
        raise ValueError("数据必须完整包含 train、validation、test 三个划分。")
    output_dir.mkdir(parents=True, exist_ok=True)
    checksums: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        target = output_dir / f"{split}.csv"
        frame[frame["split"] == split].sort_values("id").to_csv(target, index=False)
        checksums[target.name] = hashlib.sha256(target.read_bytes()).hexdigest()
    metadata = {
        "version": "dataset_v1",
        "seed": seed,
        "labels": list(EMOTIONS),
        "sources": sources,
        "rows": len(frame),
        "counts_by_split": dict(Counter(frame["split"])),
        "counts_by_language": dict(Counter(frame["language"])),
        "counts_by_label": dict(Counter(frame["label"])),
        "sha256": checksums,
        "multi_label_policy": "exclude rows with more than one Ekman label",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build fixed bilingual emotion dataset splits.")
    parser.add_argument("--goemotions-dir", type=Path, required=True)
    parser.add_argument("--chinese-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/project-data/processed/dataset_v1"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frames = [read_goemotions(args.goemotions_dir)]
    sources = ["monologg/GoEmotions-pytorch data/ekman (derived from Google GoEmotions)"]
    if args.chinese_csv:
        frames.append(read_chinese(args.chinese_csv, args.seed))
        sources.append(str(args.chinese_csv))
    write_outputs(pd.concat(frames, ignore_index=True), args.output_dir, sources, args.seed)
    print(f"Dataset written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
