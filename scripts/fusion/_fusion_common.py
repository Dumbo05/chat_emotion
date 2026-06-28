from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from emotion_app.domain import EMOTIONS, normalize_emotion


LABEL_TO_ID = {label: index for index, label in enumerate(EMOTIONS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: str | Path, rows: list[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: str | Path) -> str | None:
    candidate = Path(path)
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def empty_probs() -> dict[str, float]:
    return {label: 0.0 for label in EMOTIONS}


def normalize_probs(raw: dict | None) -> dict[str, float]:
    values = empty_probs()
    if raw:
        for label, value in raw.items():
            try:
                values[normalize_emotion(str(label))] += float(value)
            except (ValueError, TypeError):
                continue
    total = sum(max(v, 0.0) for v in values.values())
    if total <= 0:
        return empty_probs()
    return {label: max(values[label], 0.0) / total for label in EMOTIONS}


def probs_to_array(probs: dict | None) -> np.ndarray:
    normalized = normalize_probs(probs)
    return np.asarray([normalized[label] for label in EMOTIONS], dtype=np.float64)


def array_to_probs(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    total = float(arr.sum())
    if total <= 0:
        arr = np.full(len(EMOTIONS), 1.0 / len(EMOTIONS), dtype=np.float64)
    else:
        arr = arr / total
    return {label: float(arr[index]) for index, label in enumerate(EMOTIONS)}


def apply_temperature(probs: dict | None, temperature: float) -> np.ndarray:
    arr = probs_to_array(probs)
    if arr.sum() <= 0:
        return arr
    temp = max(float(temperature), 1e-6)
    logits = np.log(np.clip(arr, 1e-12, 1.0)) / temp
    logits -= logits.max()
    exp = np.exp(logits)
    return exp / exp.sum()


def weighted_fusion(row: dict, config: dict) -> np.ndarray:
    temperatures = config.get("temperatures", {})
    weights = config.get("weights", {})
    pieces: list[tuple[float, np.ndarray]] = []
    for modality in ("image", "speech", "text"):
        ok = bool(row.get(f"{modality}_ok", False))
        if not ok:
            continue
        weight = float(weights.get(modality, 0.0))
        if weight <= 0:
            continue
        probs = row.get(f"{modality}_probs", {})
        temp = float(temperatures.get(modality, 1.0))
        pieces.append((weight, apply_temperature(probs, temp)))
    if not pieces:
        return np.full(len(EMOTIONS), 1.0 / len(EMOTIONS), dtype=np.float64)
    total_weight = sum(weight for weight, _ in pieces)
    fused = sum(weight * arr for weight, arr in pieces) / total_weight
    return fused / fused.sum()


def gated_fusion(row: dict, config: dict) -> np.ndarray:
    gate = config.get("gate", {})
    image_threshold = float(gate.get("image_confidence_threshold", 0.75))
    image_probs = probs_to_array(row.get("image_probs", {}))
    if bool(row.get("image_ok", False)) and image_probs.max() >= image_threshold:
        override = dict(config)
        override["weights"] = gate.get(
            "image_high_confidence_weights",
            {"image": 0.75, "speech": 0.15, "text": 0.10},
        )
        return weighted_fusion(row, override)
    if not bool(row.get("image_ok", False)):
        override = dict(config)
        override["weights"] = gate.get(
            "image_missing_weights",
            {"image": 0.0, "speech": 0.70, "text": 0.30},
        )
        return weighted_fusion(row, override)
    return weighted_fusion(row, config)


def predict_rows(rows: list[dict], config: dict) -> tuple[list[str], list[str], list[np.ndarray]]:
    labels: list[str] = []
    predictions: list[str] = []
    probabilities: list[np.ndarray] = []
    method = config.get("method", "weighted")
    for row in rows:
        if row.get("label") not in EMOTIONS:
            continue
        fused = gated_fusion(row, config) if method == "gated" else weighted_fusion(row, config)
        labels.append(row["label"])
        predictions.append(ID_TO_LABEL[int(fused.argmax())])
        probabilities.append(fused)
    return labels, predictions, probabilities


def evaluate_predictions(labels: list[str], predictions: list[str]) -> dict:
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, predictions, average="weighted", zero_division=0),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=list(EMOTIONS),
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=list(EMOTIONS)).tolist(),
    }


def evaluate_config(rows: list[dict], config: dict) -> dict:
    labels, predictions, _ = predict_rows(rows, config)
    return evaluate_predictions(labels, predictions)


def modality_config(modality: str) -> dict:
    return {
        "method": "weighted",
        "weights": {
            "image": 1.0 if modality == "image" else 0.0,
            "speech": 1.0 if modality == "speech" else 0.0,
            "text": 1.0 if modality == "text" else 0.0,
        },
        "temperatures": {"image": 1.0, "speech": 1.0, "text": 1.0},
    }


def feature_matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for row in rows:
        if row.get("label") not in EMOTIONS:
            continue
        pieces = []
        for modality in ("image", "speech", "text"):
            arr = probs_to_array(row.get(f"{modality}_probs", {}))
            if not bool(row.get(f"{modality}_ok", False)):
                arr = np.zeros_like(arr)
            pieces.append(arr)
        xs.append(np.concatenate(pieces))
        ys.append(LABEL_TO_ID[row["label"]])
    return np.vstack(xs), np.asarray(ys, dtype=np.int64)
