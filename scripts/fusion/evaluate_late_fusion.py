from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from emotion_app.domain import EMOTIONS
from scripts.fusion._fusion_common import (
    evaluate_config,
    evaluate_predictions,
    feature_matrix,
    modality_config,
    predict_rows,
    read_jsonl,
    sha256_file,
)


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def evaluate_meta(rows: list[dict], config: dict) -> dict:
    clf = joblib.load(config["model_path"])
    x, y = feature_matrix(rows)
    pred_ids = clf.predict(x)
    labels = [EMOTIONS[int(index)] for index in y]
    preds = [EMOTIONS[int(index)] for index in pred_ids]
    return evaluate_predictions(labels, preds)


def write_confusion(path: Path, matrix: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual\\predicted", *EMOTIONS])
        for label, row in zip(EMOTIONS, matrix):
            writer.writerow([label, *row])


def report_lines(metrics: dict, single: dict, config: dict, test_probs: Path) -> list[str]:
    lines = [
        "# Multimodal Late Fusion Report",
        "",
        "## Frozen config",
        "",
        f"- Best run: `{config['best_run_id']}`",
        f"- Method: `{config['config'].get('method')}`",
        f"- Test probability file: `{test_probs}`",
        f"- Test probability SHA-256: `{sha256_file(test_probs)}`",
        "",
        "## Final test metrics",
        "",
        f"- Accuracy: **{pct(metrics['accuracy'])}**",
        f"- Macro-F1: **{pct(metrics['macro_f1'])}**",
        f"- Weighted-F1: **{pct(metrics['weighted_f1'])}**",
        "",
        "## Same-test baselines",
        "",
        "| Model | Accuracy | Macro-F1 | Weighted-F1 |",
        "|---|---:|---:|---:|",
    ]
    for name in ("image", "speech", "text"):
        item = single[name]
        lines.append(f"| {name} | {pct(item['accuracy'])} | {pct(item['macro_f1'])} | {pct(item['weighted_f1'])} |")
    lines += [
        f"| fusion | {pct(metrics['accuracy'])} | {pct(metrics['macro_f1'])} | {pct(metrics['weighted_f1'])} |",
        "",
        "## Per-class fusion metrics",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    report = metrics["classification_report"]
    for label in EMOTIONS:
        row = report[label]
        lines.append(
            f"| {label} | {pct(row['precision'])} | {pct(row['recall'])} | {pct(row['f1-score'])} | {int(row['support'])} |"
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a frozen late-fusion config on test probabilities.")
    parser.add_argument("--test-probs", type=Path, required=True)
    parser.add_argument("--fusion-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.test_probs)
    frozen = json.loads(args.fusion_config.read_text(encoding="utf-8"))
    config = frozen["config"]

    if config.get("method") == "meta_logreg":
        metrics = evaluate_meta(rows, config)
    else:
        metrics = evaluate_config(rows, config)

    single = {name: evaluate_config(rows, modality_config(name)) for name in ("image", "speech", "text")}
    final = {
        "fusion": metrics,
        "single_modality": single,
        "frozen_config": frozen,
        "test_probs": str(args.test_probs),
        "test_probs_sha256": sha256_file(args.test_probs),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "final_metrics.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    write_confusion(args.output_dir / "confusion_matrix.csv", metrics["confusion_matrix"])
    (args.output_dir / "report.md").write_text("\n".join(report_lines(metrics, single, frozen, args.test_probs)) + "\n", encoding="utf-8")
    print(json.dumps({"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"], "weighted_f1": metrics["weighted_f1"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
