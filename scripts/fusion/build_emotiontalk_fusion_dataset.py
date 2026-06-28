from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from emotion_app.domain import EMOTIONS, normalize_emotion
from scripts.fusion._fusion_common import sha256_file, write_csv_rows


AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".m4a", ".aac"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def clean_text(value: str) -> str:
    text = re.sub(r"\[(?:/)?over/?.*?\]", "", str(value))
    text = text.replace("[over/]", "").replace("[/over]", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_label(value: str) -> str | None:
    raw = str(value).strip().lower()
    aliases = {
        "angry": "anger",
        "anger": "anger",
        "disgust": "disgust",
        "fear": "fear",
        "fearful": "fear",
        "happy": "joy",
        "happiness": "joy",
        "joy": "joy",
        "sad": "sadness",
        "sadness": "sadness",
        "surprise": "surprise",
        "surprised": "surprise",
        "neutral": "neutral",
    }
    mapped = aliases.get(raw)
    return normalize_emotion(mapped) if mapped else None


def sample_id_from_file_name(file_name: str) -> str:
    return Path(str(file_name)).stem


def group_id_from_file_name(file_name: str) -> str:
    parts = Path(str(file_name)).parts
    if len(parts) >= 2:
        return "/".join(parts[:2])
    return Path(str(file_name)).stem


def index_files(root: Path, suffixes: set[str]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    if not root or not root.exists():
        return index
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            index.setdefault(path.stem, path.resolve())
    return index


def index_contains(root: Path, suffixes: set[str]) -> list[Path]:
    if not root or not root.exists():
        return []
    return [path.resolve() for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes]


def find_by_id(sample_id: str, exact_index: dict[str, Path], candidates: list[Path]) -> str:
    if sample_id in exact_index:
        return str(exact_index[sample_id])
    for path in candidates:
        if sample_id in path.stem or sample_id in path.as_posix():
            return str(path)
    return ""


def extract_frame(video_path: Path, output_path: Path, seconds: float = 0.5) -> bool:
    try:
        import cv2
    except Exception:
        return False
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    target = min(max(int(seconds * fps), 0), max(int(frame_count) - 1, 0))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), frame))


def load_text_rows(text_root: Path) -> list[dict]:
    json_root = text_root / "Text" / "json"
    if not json_root.exists():
        json_root = text_root
    rows: list[dict] = []
    for path in sorted(json_root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        label = normalize_label(payload.get("emotion_result", ""))
        text = clean_text(payload.get("content", ""))
        file_name = payload.get("file_name") or path.with_suffix(".txt").name
        sample_id = sample_id_from_file_name(file_name)
        if not sample_id or not label or not text:
            continue
        rows.append(
            {
                "sample_id": sample_id,
                "label": label,
                "text": text,
                "speaker_id": str(payload.get("speaker_id") or ""),
                "group_id": group_id_from_file_name(file_name),
                "source": "emotiontalk",
                "text_json_path": str(path.resolve()),
                "file_name": str(file_name),
            }
        )
    return rows


def split_rows(rows: list[dict], seed: int, ratios: tuple[float, float, float]) -> dict[str, list[dict]]:
    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        key = row.get("speaker_id") or row.get("group_id") or row["sample_id"]
        by_group[str(key)].append(row)
    groups = list(by_group)
    random.Random(seed).shuffle(groups)
    total = sum(len(by_group[group]) for group in groups)
    train_target = total * ratios[0]
    val_target = total * ratios[1]
    splits = {"train": [], "validation": [], "test": []}
    for group in groups:
        current_train = len(splits["train"])
        current_val = len(splits["validation"])
        if current_train < train_target:
            target = "train"
        elif current_val < val_target:
            target = "validation"
        else:
            target = "test"
        splits[target].extend(by_group[group])
    return splits


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an aligned EmotionTalk multimodal dataset manifest.")
    parser.add_argument("--text-root", type=Path, default=Path("/data/emotion-text-datasets/raw/emotiontalk_text"))
    parser.add_argument("--audio-root", type=Path, default=Path("/data/emotion-text-datasets/raw/emotiontalk_audio"))
    parser.add_argument("--image-root", type=Path, default=Path("/data/emotion-text-datasets/raw/emotiontalk_multimodal"))
    parser.add_argument("--video-root", type=Path, default=Path("/data/emotion-text-datasets/raw/emotiontalk_video"))
    parser.add_argument("--output-dir", type=Path, default=Path("/data/emotion-text-datasets/processed/dataset_fusion_v1"))
    parser.add_argument("--extract-video-frames", action="store_true")
    parser.add_argument("--frame-dir", type=Path, default=Path("/data/emotion-text-datasets/processed/dataset_fusion_v1/frames"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    rows = load_text_rows(args.text_root)
    audio_index = index_files(args.audio_root, AUDIO_SUFFIXES)
    image_index = index_files(args.image_root, IMAGE_SUFFIXES)
    video_index = index_files(args.video_root, VIDEO_SUFFIXES)
    audio_candidates = index_contains(args.audio_root, AUDIO_SUFFIXES)
    image_candidates = index_contains(args.image_root, IMAGE_SUFFIXES)
    video_candidates = index_contains(args.video_root, VIDEO_SUFFIXES)

    all_rows: list[dict] = []
    for row in rows:
        sample_id = row["sample_id"]
        audio_path = find_by_id(sample_id, audio_index, audio_candidates)
        image_path = find_by_id(sample_id, image_index, image_candidates)
        video_path = find_by_id(sample_id, video_index, video_candidates)
        if not image_path and video_path and args.extract_video_frames:
            frame_path = args.frame_dir / f"{sample_id}.jpg"
            if frame_path.is_file() or extract_frame(Path(video_path), frame_path):
                image_path = str(frame_path.resolve())
        item = {
            "sample_id": sample_id,
            "label": row["label"],
            "text": row["text"],
            "audio_path": audio_path,
            "image_path": image_path,
            "source": row["source"],
            "speaker_id": row["speaker_id"],
            "group_id": row["group_id"],
            "video_path": video_path,
            "text_json_path": row["text_json_path"],
        }
        all_rows.append(item)

    full_rows = [row for row in all_rows if row["audio_path"] and row["image_path"]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(args.output_dir / "manifest_all.csv", all_rows)
    write_csv_rows(args.output_dir / "manifest_full.csv", full_rows)

    ratios = (args.train_ratio, args.val_ratio, max(0.0, 1.0 - args.train_ratio - args.val_ratio))
    splits = split_rows(full_rows, args.seed, ratios)
    for split, split_rows_value in splits.items():
        write_csv_rows(args.output_dir / f"{split}.csv", split_rows_value)

    manifest = {
        "version": "dataset_fusion_v1",
        "source": "BAAI/Emotiontalk",
        "label_order": list(EMOTIONS),
        "seed": args.seed,
        "paths": {
            "text_root": str(args.text_root),
            "audio_root": str(args.audio_root),
            "image_root": str(args.image_root),
            "video_root": str(args.video_root),
        },
        "rows": {
            "text_rows": len(rows),
            "manifest_all": len(all_rows),
            "manifest_full": len(full_rows),
            "train": len(splits["train"]),
            "validation": len(splits["validation"]),
            "test": len(splits["test"]),
        },
        "missing": {
            "audio": sum(1 for row in all_rows if not row["audio_path"]),
            "image": sum(1 for row in all_rows if not row["image_path"]),
        },
        "counts_by_label_full": dict(Counter(row["label"] for row in full_rows)),
        "sha256": {
            name: sha256_file(args.output_dir / name)
            for name in ("manifest_all.csv", "manifest_full.csv", "train.csv", "validation.csv", "test.csv")
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest["rows"] | {"missing_audio": manifest["missing"]["audio"], "missing_image": manifest["missing"]["image"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
