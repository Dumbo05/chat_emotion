from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

RAF_LABELS = ("surprise", "fear", "disgust", "joy", "sadness", "anger", "neutral")


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits / temperature
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predicted = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predicted, labels=list(range(len(RAF_LABELS))), zero_division=0
    )
    macro = precision_recall_fscore_support(labels, predicted, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(labels, predicted, average="weighted", zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(macro[2]),
        "weighted_f1": float(weighted[2]),
        "per_class_f1": {RAF_LABELS[i]: float(f1[i]) for i in range(len(RAF_LABELS))},
        "per_class_precision": {RAF_LABELS[i]: float(precision[i]) for i in range(len(RAF_LABELS))},
        "per_class_recall": {RAF_LABELS[i]: float(recall[i]) for i in range(len(RAF_LABELS))},
        "support_per_class": {RAF_LABELS[i]: int(support[i]) for i in range(len(RAF_LABELS))},
        "confusion_matrix": confusion_matrix(labels, predicted, labels=list(range(len(RAF_LABELS)))).tolist(),
    }


def load_run(run_dir: Path) -> dict[str, Any]:
    val = np.load(run_dir / "raf_val_logits.npz", allow_pickle=True)
    test = np.load(run_dir / "raf_test_logits.npz", allow_pickle=True)
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    return {
        "run_dir": run_dir,
        "name": summary.get("run_name", run_dir.name),
        "arch": summary.get("model_arch", ""),
        "seed": summary.get("seed", ""),
        "val_logits": val["logits"],
        "val_labels": val["labels"],
        "val_paths": val["paths"],
        "test_logits": test["logits"],
        "test_labels": test["labels"],
        "test_paths": test["paths"],
    }


def assert_same_order(runs: list[dict[str, Any]]) -> None:
    first = runs[0]
    for item in runs[1:]:
        if not np.array_equal(first["val_labels"], item["val_labels"]):
            raise ValueError(f"Validation labels differ: {first['name']} vs {item['name']}")
        if not np.array_equal(first["test_labels"], item["test_labels"]):
            raise ValueError(f"Test labels differ: {first['name']} vs {item['name']}")
        if not np.array_equal(first["val_paths"], item["val_paths"]):
            raise ValueError(f"Validation sample order differs: {first['name']} vs {item['name']}")
        if not np.array_equal(first["test_paths"], item["test_paths"]):
            raise ValueError(f"Test sample order differs: {first['name']} vs {item['name']}")


def average_probs(runs: list[dict[str, Any]], indices: tuple[int, ...], temperature: float, split: str) -> np.ndarray:
    probs = [softmax(runs[i][f"{split}_logits"], temperature) for i in indices]
    return np.mean(probs, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/image/rafdb_image_v4_strong/ensemble"))
    parser.add_argument("--max-members", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--min-val-macro-f1", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = [load_run(path) for path in args.run_dirs]
    assert_same_order(runs)

    rows: list[dict[str, Any]] = []
    single_rows: list[dict[str, Any]] = []
    for index, run in enumerate(runs):
        val_probs = softmax(run["val_logits"], args.temperature)
        test_probs = softmax(run["test_logits"], args.temperature)
        row = {
            "indices": (index,),
            "members": [run["name"]],
            "val_metrics": metrics(run["val_labels"], val_probs),
            "test_metrics": metrics(run["test_labels"], test_probs),
        }
        rows.append(row)
        single_rows.append(row)

    candidate_indices = [
        int(row["indices"][0])
        for row in single_rows
        if row["val_metrics"]["macro_f1"] >= args.min_val_macro_f1
    ]
    if not candidate_indices:
        candidate_indices = list(range(len(runs)))

    max_members = min(args.max_members, len(candidate_indices))
    for size in range(2, max_members + 1):
        for indices in combinations(candidate_indices, size):
            val_probs = average_probs(runs, indices, args.temperature, "val")
            test_probs = average_probs(runs, indices, args.temperature, "test")
            rows.append(
                {
                    "indices": indices,
                    "members": [runs[i]["name"] for i in indices],
                    "val_metrics": metrics(runs[0]["val_labels"], val_probs),
                    "test_metrics": metrics(runs[0]["test_labels"], test_probs),
                }
            )

    rows.sort(key=lambda item: (item["val_metrics"]["macro_f1"], item["val_metrics"]["accuracy"]), reverse=True)
    best = rows[0]
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "ensemble_results.json", {"best": best, "all": rows, "temperature": args.temperature})

    with (args.output / "ensemble_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = [
            "rank",
            "members",
            "val_accuracy",
            "val_macro_f1",
            "test_accuracy",
            "test_macro_f1",
            "test_fear_f1",
            "test_disgust_f1",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "members": " + ".join(row["members"]),
                    "val_accuracy": row["val_metrics"]["accuracy"],
                    "val_macro_f1": row["val_metrics"]["macro_f1"],
                    "test_accuracy": row["test_metrics"]["accuracy"],
                    "test_macro_f1": row["test_metrics"]["macro_f1"],
                    "test_fear_f1": row["test_metrics"]["per_class_f1"]["fear"],
                    "test_disgust_f1": row["test_metrics"]["per_class_f1"]["disgust"],
                }
            )
    print(json.dumps(json_ready({"best": best, "output": args.output}), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
