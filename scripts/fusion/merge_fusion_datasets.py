from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from emotion_app.domain import EMOTIONS
from scripts.fusion._fusion_common import sha256_file


REQUIRED_COLUMNS = [
    "sample_id",
    "label",
    "text",
    "audio_path",
    "image_path",
    "source",
    "speaker_id",
    "group_id",
    "video_path",
    "language",
]


def read_split(root: Path, split: str, source_prefix: str) -> pd.DataFrame:
    path = root / f"{split}.csv"
    frame = pd.read_csv(path)
    for column in REQUIRED_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    frame = frame[REQUIRED_COLUMNS].copy()
    frame["sample_id"] = source_prefix + "_" + frame["sample_id"].astype(str)
    frame["label"] = frame["label"].astype(str)
    frame = frame[frame["label"].isin(EMOTIONS)].copy()
    frame = frame[(frame["audio_path"].fillna("") != "") & (frame["image_path"].fillna("") != "")]
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge aligned multimodal fusion datasets by split.")
    parser.add_argument("--emotiontalk-dir", type=Path, required=True)
    parser.add_argument("--meld-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/data/emotion-text-datasets/processed/dataset_fusion_v2_emotiontalk_meld_7class"))
    parser.add_argument("--meld-disgust-only", action="store_true", help="Only add MELD disgust rows; default adds all MELD labels.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_split: dict[str, pd.DataFrame] = {}
    for split in ("train", "validation", "test"):
        emotiontalk = read_split(args.emotiontalk_dir, split, "emotiontalk")
        meld = read_split(args.meld_dir, split, "meld")
        if args.meld_disgust_only:
            meld = meld[meld["label"] == "disgust"].copy()
        merged = pd.concat([emotiontalk, meld], ignore_index=True)
        merged.to_csv(args.output_dir / f"{split}.csv", index=False)
        rows_by_split[split] = merged

    all_rows = pd.concat(list(rows_by_split.values()), ignore_index=True)
    all_rows.to_csv(args.output_dir / "manifest_full.csv", index=False)
    all_rows.to_csv(args.output_dir / "manifest_all.csv", index=False)

    manifest = {
        "version": "dataset_fusion_v2_emotiontalk_meld_7class",
        "sources": {
            "emotiontalk": str(args.emotiontalk_dir),
            "meld": str(args.meld_dir),
            "meld_disgust_only": args.meld_disgust_only,
        },
        "label_order": list(EMOTIONS),
        "rows": {split: int(len(rows)) for split, rows in rows_by_split.items()},
        "counts_by_label_full": dict(Counter(all_rows["label"])),
        "counts_by_source_full": dict(Counter(all_rows["source"])),
        "sha256": {
            name: sha256_file(args.output_dir / name)
            for name in ("manifest_all.csv", "manifest_full.csv", "train.csv", "validation.csv", "test.csv")
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": manifest["rows"], "counts": manifest["counts_by_label_full"], "sources": manifest["counts_by_source_full"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
