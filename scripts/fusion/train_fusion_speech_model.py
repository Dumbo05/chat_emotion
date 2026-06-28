from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from emotion_app.audio_features import _read_audio
from emotion_app.domain import EMOTIONS
from scripts.fusion._fusion_common import sha256_file, write_jsonl


LABEL_TO_ID = {label: index for index, label in enumerate(EMOTIONS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


def frame_signal(signal: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if signal.size < frame_length:
        signal = np.pad(signal, (0, frame_length - signal.size))
    frame_count = 1 + max(0, (signal.size - frame_length) // hop_length)
    frames = np.lib.stride_tricks.sliding_window_view(signal, frame_length)[::hop_length]
    return frames[:frame_count]


def summarize(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        float(np.mean(values)),
        float(np.std(values)),
        float(np.percentile(values, 10)),
        float(np.percentile(values, 90)),
    ]


def extract_features(path: str | Path, max_seconds: float = 12.0, target_rate: int = 16000) -> np.ndarray:
    signal = _read_audio(path, target_rate=target_rate).astype(np.float32)
    max_len = int(max_seconds * target_rate)
    signal = signal[:max_len]
    if signal.size == 0:
        return np.zeros(80, dtype=np.float32)
    signal = signal - float(signal.mean())
    signal = signal / (float(signal.std()) + 1e-6)

    frame_length = int(0.025 * target_rate)
    hop_length = int(0.010 * target_rate)
    frames = frame_signal(signal, frame_length, hop_length)
    window = np.hanning(frame_length).astype(np.float32)
    spec = np.abs(np.fft.rfft(frames * window[None, :], axis=1)) + 1e-8
    power = spec ** 2
    freqs = np.fft.rfftfreq(frame_length, 1.0 / target_rate)
    energy = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-8)
    zcr = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1)
    centroid = np.sum(power * freqs[None, :], axis=1) / np.sum(power, axis=1)
    bandwidth = np.sqrt(np.sum(power * (freqs[None, :] - centroid[:, None]) ** 2, axis=1) / np.sum(power, axis=1))
    cumulative = np.cumsum(power, axis=1)
    rolloff_index = np.argmax(cumulative >= 0.85 * cumulative[:, -1:], axis=1)
    rolloff = freqs[rolloff_index]

    features: list[float] = []
    for values in (energy, zcr, centroid, bandwidth, rolloff):
        features.extend(summarize(values))

    bands = [
        (0, 250),
        (250, 500),
        (500, 1000),
        (1000, 2000),
        (2000, 4000),
        (4000, 8000),
    ]
    total_power = np.sum(power, axis=1) + 1e-8
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        ratio = np.sum(power[:, mask], axis=1) / total_power
        features.extend(summarize(ratio))

    # Coarse log spectrum shape. This is cheap and surprisingly useful for
    # emotional speech when a heavy pretrained encoder is not available.
    bins = np.array_split(np.log(power), 36, axis=1)
    for item in bins:
        features.append(float(np.mean(item)))

    return np.asarray(features, dtype=np.float32)


def read_split(data_dir: Path, split: str) -> pd.DataFrame:
    frame = pd.read_csv(data_dir / f"{split}.csv")
    frame = frame[frame["label"].isin(EMOTIONS)].copy()
    frame = frame[frame["audio_path"].map(lambda x: Path(str(x)).is_file())].copy()
    return frame.reset_index(drop=True)


def cached_features(frame: pd.DataFrame, cache_path: Path, max_seconds: float) -> np.ndarray:
    if cache_path.is_file():
        return np.load(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in tqdm(frame["audio_path"], desc=f"features {cache_path.name}"):
        rows.append(extract_features(path, max_seconds=max_seconds))
    array = np.vstack(rows)
    np.save(cache_path, array)
    return array


def build_classifier(kind: str):
    if kind == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, n_jobs=-1),
        )
    if kind == "rbf_svm":
        return make_pipeline(
            StandardScaler(),
            SVC(C=3.0, gamma="scale", class_weight="balanced", probability=True),
        )
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        )
    raise ValueError(f"Unsupported classifier: {kind}")


def evaluate(model, x: np.ndarray, y: np.ndarray, sample_ids: list[str]) -> tuple[dict, list[dict]]:
    probs = model.predict_proba(x)
    classes = list(model.classes_)
    full_probs = np.zeros((len(x), len(EMOTIONS)), dtype=np.float64)
    for column, class_id in enumerate(classes):
        full_probs[:, int(class_id)] = probs[:, column]
    predictions = full_probs.argmax(axis=1)
    metrics = {
        "accuracy": accuracy_score(y, predictions),
        "macro_f1": f1_score(y, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(y, predictions, average="weighted", zero_division=0),
        "classification_report": classification_report(
            y,
            predictions,
            labels=list(range(len(EMOTIONS))),
            target_names=list(EMOTIONS),
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y, predictions, labels=list(range(len(EMOTIONS)))).tolist(),
    }
    rows = []
    for sample_id, label_id, prob in zip(sample_ids, y, full_probs):
        rows.append(
            {
                "sample_id": sample_id,
                "label": ID_TO_LABEL[int(label_id)],
                "speech_probs": {emotion: float(prob[index]) for index, emotion in enumerate(EMOTIONS)},
                "speech_ok": True,
                "speech_pred": ID_TO_LABEL[int(prob.argmax())],
                "speech_confidence": float(prob.max()),
            }
        )
    return metrics, rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a lightweight speech classifier for fusion datasets.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--classifier", choices=["logreg", "rbf_svm", "rf"], default="logreg")
    parser.add_argument("--max-seconds", type=float, default=12.0)
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = {split: read_split(args.data_dir, split) for split in ("train", "validation", "test")}
    if args.refresh_cache:
        for path in args.output_dir.glob("*_features.npy"):
            path.unlink()

    features = {
        split: cached_features(frame, args.output_dir / f"{split}_features.npy", args.max_seconds)
        for split, frame in frames.items()
    }
    labels = {
        split: frames[split]["label"].map(LABEL_TO_ID).to_numpy(dtype=np.int64)
        for split in frames
    }
    model = build_classifier(args.classifier)
    model.fit(features["train"], labels["train"])

    val_metrics, val_rows = evaluate(model, features["validation"], labels["validation"], frames["validation"]["sample_id"].astype(str).tolist())
    test_metrics, test_rows = evaluate(model, features["test"], labels["test"], frames["test"]["sample_id"].astype(str).tolist())
    model_path = args.output_dir / "speech_model.joblib"
    joblib.dump(model, model_path)
    write_jsonl(args.output_dir / "val_speech_probs.jsonl", val_rows)
    write_jsonl(args.output_dir / "test_speech_probs.jsonl", test_rows)

    metrics = {
        "classifier": args.classifier,
        "label_order": list(EMOTIONS),
        "validation": val_metrics,
        "test": test_metrics,
        "data": {split: len(frame) for split, frame in frames.items()},
        "sha256": {"speech_model.joblib": sha256_file(model_path)},
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"validation_accuracy": val_metrics["accuracy"], "validation_macro_f1": val_metrics["macro_f1"], "test_accuracy": test_metrics["accuracy"], "test_macro_f1": test_metrics["macro_f1"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
