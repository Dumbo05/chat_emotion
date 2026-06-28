from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from emotion_app.domain import EMOTIONS
from scripts.fusion._fusion_common import (
    evaluate_config,
    evaluate_predictions,
    feature_matrix,
    read_jsonl,
    sha256_file,
)


def score_key(record: dict) -> tuple[float, float, float]:
    metrics = record["validation"]
    return (
        float(metrics["accuracy"]),
        float(metrics["macro_f1"]),
        float(metrics["weighted_f1"]),
    )


def weight_grid() -> list[dict[str, float]]:
    configs: list[dict[str, float]] = []
    for image in np.arange(0.45, 0.801, 0.05):
        for speech in np.arange(0.10, 0.401, 0.05):
            for text in np.arange(0.05, 0.251, 0.05):
                total = float(image + speech + text)
                configs.append(
                    {
                        "image": round(float(image / total), 6),
                        "speech": round(float(speech / total), 6),
                        "text": round(float(text / total), 6),
                    }
                )
    unique = {tuple(sorted(item.items())): item for item in configs}
    return list(unique.values())


def temperature_grid() -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for text, speech, image in product((1.0, 1.5, 2.0), (0.8, 1.0, 1.2), (0.8, 1.0, 1.2)):
        rows.append({"text": text, "speech": speech, "image": image})
    return rows


def compact_metrics(metrics: dict) -> dict:
    return {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune late-fusion strategy on validation probabilities.")
    parser.add_argument("--val-probs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-probs", type=Path, default=None)
    parser.add_argument("--model-hash-path", type=Path, action="append", default=[])
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    val_rows = read_jsonl(args.val_probs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    for modality in ("image", "speech", "text"):
        config = {
            "method": "weighted",
            "weights": {
                "image": 1.0 if modality == "image" else 0.0,
                "speech": 1.0 if modality == "speech" else 0.0,
                "text": 1.0 if modality == "text" else 0.0,
            },
            "temperatures": {"image": 1.0, "speech": 1.0, "text": 1.0},
        }
        records.append(
            {
                "run_id": f"single_{modality}",
                "config": config,
                "validation": compact_metrics(evaluate_config(val_rows, config)),
            }
        )

    for index, (weights, temperatures) in enumerate(product(weight_grid(), temperature_grid())):
        config = {
            "method": "weighted",
            "weights": weights,
            "temperatures": temperatures,
        }
        metrics = compact_metrics(evaluate_config(val_rows, config))
        records.append({"run_id": f"weighted_{index:05d}", "config": config, "validation": metrics})

        gated = {
            "method": "gated",
            "weights": weights,
            "temperatures": temperatures,
            "gate": {
                "image_confidence_threshold": 0.75,
                "image_high_confidence_weights": {"image": 0.75, "speech": 0.15, "text": 0.10},
                "image_missing_weights": {"image": 0.0, "speech": 0.70, "text": 0.30},
            },
        }
        gated_metrics = compact_metrics(evaluate_config(val_rows, gated))
        records.append({"run_id": f"gated_{index:05d}", "config": gated, "validation": gated_metrics})

    if args.train_probs and args.train_probs.is_file():
        train_rows = read_jsonl(args.train_probs)
        x_train, y_train = feature_matrix(train_rows)
        x_val, y_val = feature_matrix(val_rows)
        for c in (0.1, 0.3, 1.0, 3.0):
            clf = LogisticRegression(C=c, max_iter=1000, class_weight="balanced", multi_class="auto")
            clf.fit(x_train, y_train)
            pred_ids = clf.predict(x_val)
            preds = [EMOTIONS[int(index)] for index in pred_ids]
            labels = [EMOTIONS[int(index)] for index in y_val]
            model_path = args.output_dir / f"meta_logreg_c{str(c).replace('.', 'p')}.joblib"
            joblib.dump(clf, model_path)
            config = {
                "method": "meta_logreg",
                "model_path": str(model_path),
                "C": c,
                "feature_order": ["image_probs", "speech_probs", "text_probs"],
            }
            records.append(
                {
                    "run_id": f"meta_logreg_c{c}",
                    "config": config,
                    "validation": compact_metrics(evaluate_predictions(labels, preds)),
                }
            )

    records.sort(key=score_key, reverse=True)
    best = records[0]
    model_hashes = {str(path): sha256_file(path) for path in args.model_hash_path}
    fusion_config = {
        "version": "dataset_fusion_v1",
        "selection_metric": "validation accuracy, then macro_f1, then weighted_f1",
        "label_order": list(EMOTIONS),
        "best_run_id": best["run_id"],
        "config": best["config"],
        "validation": best["validation"],
        "source_files": {
            "val_probs": str(args.val_probs),
            "train_probs": str(args.train_probs) if args.train_probs else None,
            "val_probs_sha256": sha256_file(args.val_probs),
            "train_probs_sha256": sha256_file(args.train_probs) if args.train_probs else None,
        },
        "model_hashes": model_hashes,
    }
    (args.output_dir / "fusion_config.json").write_text(json.dumps(fusion_config, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "best": fusion_config,
        "top": records[: args.top_k],
        "candidate_count": len(records),
    }
    (args.output_dir / "validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_run_id": best["run_id"], **best["validation"], "candidate_count": len(records)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
