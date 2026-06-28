from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_language_split(input_dir: Path, output_root: Path, language: str) -> dict:
    output_dir = output_root / f"dataset_v4_{language}_only"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "version": f"dataset_v4_{language}_only",
        "source": str(input_dir),
        "language": language,
        "splits": {},
        "counts_by_label": {},
        "sha256": {},
    }

    all_rows: list[pd.DataFrame] = []
    for split in ("train", "validation", "test"):
        frame = pd.read_csv(input_dir / f"{split}.csv")
        frame = frame[frame["language"].astype(str) == language].copy()
        frame = frame.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
        frame.to_csv(output_dir / f"{split}.csv", index=False)
        all_rows.append(frame.assign(split=split))
        manifest["splits"][split] = int(len(frame))
        manifest["sha256"][f"{split}.csv"] = sha256_file(output_dir / f"{split}.csv")

    combined = pd.concat(all_rows, ignore_index=True)
    manifest["counts_by_label"] = {
        split: dict(Counter(frame["label"]))
        for split, frame in zip(("train", "validation", "test"), all_rows)
    }
    manifest["counts_all"] = dict(Counter(combined["label"]))
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Create language-specific text emotion datasets.")
    parser.add_argument("--input-dir", type=Path, default=Path("datasets/project-data/processed/dataset_v1"))
    parser.add_argument("--output-root", type=Path, default=Path("datasets/project-data/processed"))
    args = parser.parse_args()

    manifests = [
        write_language_split(args.input_dir, args.output_root, "en"),
        write_language_split(args.input_dir, args.output_root, "zh"),
    ]
    print(json.dumps(manifests, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
