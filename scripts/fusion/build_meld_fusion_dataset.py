from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from emotion_app.domain import EMOTIONS, normalize_emotion
from scripts.fusion._fusion_common import sha256_file, write_csv_rows


SPLIT_ALIASES = {
    "train": ("train", "training"),
    "validation": ("dev", "val", "validation"),
    "test": ("test",),
}


def find_csv(root: Path, split: str) -> Path:
    aliases = SPLIT_ALIASES[split]
    candidates = []
    for path in root.rglob("*.csv"):
        name = path.name.lower()
        if "sent_emo" not in name and "emotion" not in name:
            continue
        if any(alias in name for alias in aliases):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"Cannot find MELD {split} CSV under {root}")
    return sorted(candidates, key=lambda p: (len(p.parts), str(p)))[0]


def find_video(root: Path, dialogue_id: int, utterance_id: int) -> Path | None:
    names = [
        f"dia{dialogue_id}_utt{utterance_id}.mp4",
        f"dia{dialogue_id}_utt{utterance_id}.avi",
        f"dia{dialogue_id}_utt{utterance_id}.mkv",
    ]
    for name in names:
        matches = list(root.rglob(name))
        if matches:
            return matches[0].resolve()
    return None


def clean_label(value: str) -> str | None:
    try:
        return normalize_emotion(str(value))
    except ValueError:
        return None


def extract_frame(video_path: Path, frame_path: Path, seconds: float) -> bool:
    if frame_path.is_file():
        return True
    try:
        import cv2
    except Exception:
        return False
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target = min(max(int(seconds * fps), 0), max(total - 1, 0))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return False
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(frame_path), frame))


def extract_audio(video_path: Path, audio_path: Path) -> bool:
    if audio_path.is_file():
        return True
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-loglevel",
        "error",
        str(audio_path),
    ]
    return subprocess.run(cmd, check=False).returncode == 0 and audio_path.is_file()


def build_split(args: argparse.Namespace, split: str) -> list[dict]:
    csv_path = find_csv(args.meld_root, split)
    frame_dir = args.output_dir / "frames" / split
    audio_dir = args.output_dir / "audio" / split
    frame = pd.read_csv(csv_path)
    rows: list[dict] = []
    for raw in frame.to_dict("records"):
        label = clean_label(raw.get("Emotion", raw.get("emotion", "")))
        if label not in EMOTIONS:
            continue
        dialogue_id = int(raw.get("Dialogue_ID", raw.get("dialogue_id")))
        utterance_id = int(raw.get("Utterance_ID", raw.get("utterance_id")))
        sample_id = f"meld_{split}_dia{dialogue_id}_utt{utterance_id}"
        video_path = find_video(args.meld_root, dialogue_id, utterance_id)
        if not video_path:
            continue
        image_path = frame_dir / f"{sample_id}.jpg"
        audio_path = audio_dir / f"{sample_id}.wav"
        if args.extract_media:
            if not extract_frame(video_path, image_path, args.frame_second):
                image_path = Path("")
            if not extract_audio(video_path, audio_path):
                audio_path = Path("")
        rows.append(
            {
                "sample_id": sample_id,
                "label": label,
                "text": str(raw.get("Utterance", raw.get("utterance", ""))).strip(),
                "audio_path": str(audio_path.resolve()) if audio_path else "",
                "image_path": str(image_path.resolve()) if image_path else "",
                "source": "meld",
                "speaker_id": str(raw.get("Speaker", raw.get("speaker", ""))),
                "group_id": f"meld_{split}_dia{dialogue_id}",
                "video_path": str(video_path),
                "language": "en",
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a 7-class multimodal MELD fusion dataset.")
    parser.add_argument("--meld-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/data/emotion-text-datasets/processed/dataset_meld_fusion_7class"))
    parser.add_argument("--extract-media", action="store_true")
    parser.add_argument("--frame-second", type=float, default=0.5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_rows = {split: build_split(args, split) for split in ("train", "validation", "test")}
    all_rows = [row for rows in split_rows.values() for row in rows]
    full_rows = [row for row in all_rows if row["audio_path"] and row["image_path"]]

    for split, rows in split_rows.items():
        write_csv_rows(args.output_dir / f"{split}.csv", rows)
    write_csv_rows(args.output_dir / "manifest_all.csv", all_rows)
    write_csv_rows(args.output_dir / "manifest_full.csv", full_rows)

    manifest = {
        "version": "dataset_meld_fusion_7class",
        "source": "MELD",
        "label_order": list(EMOTIONS),
        "rows": {split: len(rows) for split, rows in split_rows.items()},
        "manifest_all": len(all_rows),
        "manifest_full": len(full_rows),
        "counts_by_label_full": dict(Counter(row["label"] for row in full_rows)),
        "sha256": {
            name: sha256_file(args.output_dir / name)
            for name in ("manifest_all.csv", "manifest_full.csv", "train.csv", "validation.csv", "test.csv")
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": manifest["rows"], "manifest_full": len(full_rows), "counts": manifest["counts_by_label_full"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
