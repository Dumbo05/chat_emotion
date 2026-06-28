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


def context_text(group: pd.DataFrame, index: int, before: int, after: int) -> str:
    rows = group.reset_index(drop=True)
    current = rows.iloc[index]
    start = max(0, index - before)
    end = min(len(rows), index + after + 1)
    pieces = []
    for pos in range(start, end):
        row = rows.iloc[pos]
        speaker = str(row.get("speaker_id", row.get("Speaker", ""))).strip() or "speaker"
        utterance = str(row["text"]).strip()
        marker = "CURRENT" if pos == index else "CTX"
        pieces.append(f"[{marker}] {speaker}: {utterance}")
    return " </s> ".join(pieces)


def build_split(path: Path, before: int, after: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["label"].isin(EMOTIONS)].copy()
    if "language" not in frame.columns:
        frame["language"] = "en"
    frame["_order"] = frame["sample_id"].astype(str).str.extract(r"_utt(\d+)$")[0].fillna("0").astype(int)
    rows = []
    for _, group in frame.sort_values(["group_id", "_order"]).groupby("group_id", sort=False):
        group = group.reset_index(drop=True)
        for index in range(len(group)):
            row = group.iloc[index].to_dict()
            row["original_text"] = row["text"]
            row["text"] = context_text(group, index, before, after)
            rows.append(row)
    result = pd.DataFrame(rows)
    result = result.drop(columns=[column for column in ["_order"] if column in result.columns])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Build MELD context-window text dataset for utterance emotion classification.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--before", type=int, default=3)
    parser.add_argument("--after", type=int, default=1)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ("train", "validation", "test"):
        frame = build_split(args.input_dir / f"{split}.csv", args.before, args.after)
        frame.to_csv(args.output_dir / f"{split}.csv", index=False)
        counts[split] = {
            "rows": len(frame),
            "labels": dict(Counter(frame["label"])),
            "mean_chars": float(frame["text"].astype(str).str.len().mean()),
        }
    manifest = {
        "version": f"meld_context_b{args.before}_a{args.after}",
        "source_dir": str(args.input_dir),
        "context_before": args.before,
        "context_after": args.after,
        "label_order": list(EMOTIONS),
        "counts": counts,
        "sha256": {
            name: sha256_file(args.output_dir / name)
            for name in ("train.csv", "validation.csv", "test.csv")
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"version": manifest["version"], "counts": counts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
