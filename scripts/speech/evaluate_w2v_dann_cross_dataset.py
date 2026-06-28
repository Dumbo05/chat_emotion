from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from emotion_app.domain import EMOTION_LABELS_ZH, EMOTIONS  # noqa: E402
from emotion_app.recognizers.w2v_dann import Wav2VecDANNPredictor  # noqa: E402


AUDIO_SUFFIXES = {".wav", ".WAV", ".flac", ".FLAC", ".mp3", ".MP3"}

CREMA_LABELS = {
    "ANG": "anger",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "joy",
    "NEU": "neutral",
    "SAD": "sadness",
}

EMODB_LABELS = {
    "W": "anger",
    "E": "disgust",
    "A": "fear",
    "F": "joy",
    "N": "neutral",
    "T": "sadness",
    # L = boredom. The deployed app has no boredom class, so it is skipped by default.
}

TESS_LABELS = {
    "angry": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "joy",
    "neutral": "neutral",
    "pleasant_surprise": "surprise",
    "ps": "surprise",
    "sad": "sadness",
}


def infer_dataset(path: Path) -> str:
    lowered = str(path).lower()
    if "crema" in lowered:
        return "crema-d"
    if "emodb" in lowered or "berlin" in lowered:
        return "emodb"
    if "tess" in lowered:
        return "tess"
    return "auto"


def label_crema(path: Path) -> str | None:
    parts = path.stem.split("_")
    if len(parts) >= 3:
        return CREMA_LABELS.get(parts[2].upper())
    return None


def label_emodb(path: Path) -> str | None:
    # Berlin EmoDB names use the sixth character as emotion code, e.g. 03a01Fa -> F.
    stem = path.stem
    if len(stem) >= 6:
        return EMODB_LABELS.get(stem[5].upper())
    return None


def label_tess(path: Path) -> str | None:
    candidates = [path.stem.lower(), path.parent.name.lower()]
    for text in candidates:
        normalized = text.replace("-", "_").replace(" ", "_")
        for key, emotion in TESS_LABELS.items():
            if key in normalized:
                return emotion
    return None


def infer_label(path: Path, dataset: str) -> str | None:
    if dataset == "crema-d":
        return label_crema(path)
    if dataset == "emodb":
        return label_emodb(path)
    if dataset == "tess":
        return label_tess(path)
    return label_crema(path) or label_emodb(path) or label_tess(path)


def iter_items(dataset_path: Path, dataset: str, limit: int | None, max_per_class: int | None) -> list[tuple[Path, str]]:
    items: list[tuple[Path, str]] = []
    skipped = Counter()
    kept_by_class = Counter()
    for path in sorted(dataset_path.rglob("*")):
        if not path.is_file() or path.suffix not in AUDIO_SUFFIXES:
            continue
        label = infer_label(path, dataset)
        if label is None:
            skipped["unmapped_label"] += 1
            continue
        if max_per_class and kept_by_class[label] >= max_per_class:
            skipped["class_limit"] += 1
            continue
        items.append((path, label))
        kept_by_class[label] += 1
        if limit and len(items) >= limit:
            break
    if skipped:
        print("Skipped:", dict(skipped))
    return items


def safe_divide(a: int, b: int) -> float:
    return float(a / b) if b else 0.0


def build_summary(y_true: list[str], y_pred: list[str]) -> dict:
    labels = [label for label in EMOTIONS if label in set(y_true) or label in set(y_pred)]
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    per_class = {}
    for label in labels:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == label and b == label)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != label and b == label)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == label and b != label)
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        support = sum(1 for a in y_true if a == label)
        per_class[label] = {
            "zh": EMOTION_LABELS_ZH[label],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    weights = np.array([per_class[label]["support"] for label in labels], dtype=np.float64)
    f1s = np.array([per_class[label]["f1"] for label in labels], dtype=np.float64)
    recalls = np.array([per_class[label]["recall"] for label in labels], dtype=np.float64)
    weighted_f1 = float((f1s * weights).sum() / weights.sum()) if weights.sum() else 0.0
    weighted_recall = float((recalls * weights).sum() / weights.sum()) if weights.sum() else 0.0
    return {
        "n": len(y_true),
        "accuracy": safe_divide(correct, len(y_true)),
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "labels": labels,
        "per_class": per_class,
    }


def save_confusion_matrix(out_png: Path, y_true: list[str], y_pred: list[str], labels: list[str]) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    display_labels = [EMOTION_LABELS_ZH[label] for label in labels]
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    ConfusionMatrixDisplay(matrix, display_labels=display_labels).plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )
    ax.set_title("Wav2Vec2-DANN cross-dataset confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True, type=Path)
    parser.add_argument("--dataset", default=None, choices=["auto", "crema-d", "emodb", "tess"])
    parser.add_argument("--model-dir", default=ROOT / "models" / "speech" / "w2v_dann", type=Path)
    parser.add_argument("--output-dir", default=ROOT / "server-results" / "w2v_dann_cross_dataset", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--no-confusion-matrix", action="store_true")
    args = parser.parse_args()

    dataset_path = args.dataset_path.resolve()
    dataset = args.dataset or infer_dataset(dataset_path)
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    items = iter_items(dataset_path, dataset, args.limit, args.max_per_class)
    if not items:
        raise SystemExit(f"No labeled audio files found in {dataset_path}")

    predictor = Wav2VecDANNPredictor(args.model_dir)
    if not predictor.available:
        raise SystemExit(f"Missing Wav2Vec2-DANN ONNX model: {args.model_dir}")

    rows = []
    y_true = []
    y_pred = []
    for index, (path, label) in enumerate(items, start=1):
        result = predictor.predict(path)
        rows.append(
            {
                "path": str(path),
                "true": label,
                "true_zh": EMOTION_LABELS_ZH[label],
                "pred": result.emotion,
                "pred_zh": EMOTION_LABELS_ZH[result.emotion],
                "confidence": f"{result.confidence:.6f}",
                "correct": str(label == result.emotion),
            }
        )
        y_true.append(label)
        y_pred.append(result.emotion)
        if index == 1 or index % 100 == 0 or index == len(items):
            print(f"[{index}/{len(items)}] accuracy={sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true):.4f}")

    summary = build_summary(y_true, y_pred)
    stem = dataset.replace("-", "_")
    if args.limit:
        stem += f"_n{args.limit}"
    if args.max_per_class:
        stem += f"_perclass{args.max_per_class}"
    csv_path = out_dir / f"{stem}_predictions.csv"
    json_path = out_dir / f"{stem}_summary.json"
    png_path = out_dir / f"{stem}_confusion_matrix.png"

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.no_confusion_matrix:
        save_confusion_matrix(png_path, y_true, y_pred, summary["labels"])

    print("=" * 60)
    print(f"Dataset: {dataset}")
    print(f"Samples: {summary['n']}")
    print(f"Accuracy: {summary['accuracy'] * 100:.2f}%")
    print(f"Weighted Recall: {summary['weighted_recall'] * 100:.2f}%")
    print(f"Weighted F1: {summary['weighted_f1'] * 100:.2f}%")
    print(f"Predictions: {csv_path}")
    print(f"Summary: {json_path}")
    if not args.no_confusion_matrix:
        print(f"Confusion matrix: {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
