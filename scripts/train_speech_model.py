from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from emotion_app.audio_features import extract_audio_features, select_speaker_reduced_features
from emotion_app.domain import EMOTIONS

LABELS = list(EMOTIONS)
COMMON_LABELS = [label for label in LABELS if label != "surprise"]
TESS_LABELS = {
    "angry": "anger", "disgust": "disgust", "fear": "fear",
    "happy": "joy", "neutral": "neutral", "sad": "sadness",
    "pleasant_surprise": "surprise", "pleasant_surprised": "surprise",
}
CREMA_LABELS = {
    "ANG": "anger", "DIS": "disgust", "FEA": "fear",
    "HAP": "joy", "NEU": "neutral", "SAD": "sadness",
}
# Berlin EmoDB filename position 6:
# W=anger, E=disgust, A=anxiety/fear, F=joy, T=sadness, N=neutral.
# L=boredom is deliberately excluded rather than forced into a wrong class.
EMODB_LABELS = {
    "W": "anger", "E": "disgust", "A": "fear",
    "F": "joy", "T": "sadness", "N": "neutral",
}


@dataclass(frozen=True)
class Sample:
    path: Path
    label: str
    speaker: str
    dataset: str
    utterance: str

    @property
    def cache_key(self) -> str:
        return f"{self.dataset}|{self.speaker}|{self.label}|{self.path.resolve()}"


def is_riff_wav(path: Path) -> bool:
    try:
        with path.open("rb") as source:
            return source.read(4) == b"RIFF"
    except OSError:
        return False


def discover_tess(root: Path) -> list[Sample]:
    emotion_dirs = []
    for directory in root.rglob("*"):
        if not directory.is_dir():
            continue
        name = directory.name.lower()
        key = name.split("_", 1)[1] if "_" in name else name
        if key in TESS_LABELS:
            emotion_dirs.append(directory)
    if not emotion_dirs:
        raise ValueError(f"TESS 中没有找到情绪文件夹：{root}")
    min_depth = min(len(path.relative_to(root).parts) for path in emotion_dirs)
    emotion_dirs = [
        path for path in emotion_dirs
        if len(path.relative_to(root).parts) == min_depth
    ]
    samples = []
    for directory in sorted(emotion_dirs):
        key = directory.name.lower().split("_", 1)[1]
        for path in sorted(directory.glob("*.wav")):
            if not is_riff_wav(path):
                print(f"跳过非 RIFF 的伪 WAV：{path}")
                continue
            parts = path.stem.split("_")
            raw_speaker = parts[0].upper() if parts else "UNKNOWN"
            raw_speaker = "OAF" if raw_speaker == "OA" else raw_speaker
            word = parts[1].lower() if len(parts) >= 3 else path.stem.lower()
            samples.append(Sample(
                path, TESS_LABELS[key], f"TESS:{raw_speaker}", "TESS", word
            ))
    return samples


def discover_crema(root: Path) -> list[Sample]:
    samples = []
    for path in sorted(root.rglob("*.wav")):
        parts = path.stem.split("_")
        if len(parts) < 3 or parts[2].upper() not in CREMA_LABELS:
            continue
        if not is_riff_wav(path):
            print(f"跳过非 RIFF WAV：{path}")
            continue
        samples.append(Sample(
            path, CREMA_LABELS[parts[2].upper()],
            f"CREMA:{parts[0]}", "CREMA-D", parts[1],
        ))
    if not samples:
        raise ValueError(f"CREMA-D 中没有找到有效音频：{root}")
    return samples


def discover_emodb(root: Path) -> list[Sample]:
    samples = []
    excluded_boredom = 0
    for path in sorted(root.rglob("*.wav")):
        stem = path.stem
        if len(stem) < 6:
            continue
        emotion_code = stem[5].upper()
        if emotion_code == "L":
            excluded_boredom += 1
            continue
        if emotion_code not in EMODB_LABELS or not is_riff_wav(path):
            continue
        samples.append(Sample(
            path, EMODB_LABELS[emotion_code],
            f"EMODB:{stem[:2]}", "EmoDB", stem[2:5],
        ))
    if not samples:
        raise ValueError(f"EmoDB 中没有找到有效音频：{root}")
    print(f"EmoDB：排除无法映射到七类的 boredom 音频 {excluded_boredom} 条")
    return samples


def make_model(c_value: float, seed: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            C=c_value, kernel="rbf", probability=True,
            class_weight="balanced", random_state=seed,
        )),
    ])


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(
            y_true, y_pred, labels=LABELS, average="macro", zero_division=0
        )),
        "recall_macro": float(recall_score(
            y_true, y_pred, labels=LABELS, average="macro", zero_division=0
        )),
        "f1_macro": float(f1_score(
            y_true, y_pred, labels=LABELS, average="macro", zero_division=0
        )),
        "f1_weighted": float(f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        )),
        "classification_report": classification_report(
            y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=LABELS
        ).tolist(),
    }


def distribution(values: np.ndarray) -> dict[str, int]:
    counts = Counter(values.tolist())
    return {label: int(counts.get(label, 0)) for label in LABELS}


def main() -> None:
    parser = argparse.ArgumentParser(description="训练多数据集语音情感识别模型")
    parser.add_argument("--tess-dir", type=Path, default=PROJECT_ROOT / "datasets/TESS")
    parser.add_argument("--crema-dir", type=Path, default=PROJECT_ROOT / "datasets/CREMA-D")
    parser.add_argument("--emodb-dir", type=Path, default=PROJECT_ROOT / "datasets/EmoDB")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "models" / "speech")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = (
        discover_tess(args.tess_dir)
        + discover_crema(args.crema_dir)
        + discover_emodb(args.emodb_dir)
    )
    keys = np.asarray([sample.cache_key for sample in samples])
    cache_path = args.output_dir / "multidataset_features.npz"

    use_cache = False
    if cache_path.is_file() and not args.rebuild_cache:
        cached = np.load(cache_path, allow_pickle=True)
        use_cache = np.array_equal(cached["keys"].astype(str), keys.astype(str))
        if use_cache:
            X = cached["X"]
            y = cached["y"].astype(str)
            speakers = cached["speakers"].astype(str)
            datasets = cached["datasets"].astype(str)
            print(f"复用多数据集特征缓存：{cache_path}（{len(y)} 条）")
        else:
            print("数据清单已变化，自动重建特征缓存。")

    if not use_cache:
        features = []
        for index, sample in enumerate(samples, 1):
            features.append(extract_audio_features(sample.path))
            if index % 100 == 0 or index == len(samples):
                print(f"提取音频特征：{index}/{len(samples)}", flush=True)
        X = np.asarray(features)
        y = np.asarray([sample.label for sample in samples])
        speakers = np.asarray([sample.speaker for sample in samples])
        datasets = np.asarray([sample.dataset for sample in samples])
        np.savez_compressed(
            cache_path, X=X, y=y, speakers=speakers,
            datasets=datasets, keys=keys,
        )

    X_model = select_speaker_reduced_features(X)

    common_speakers = np.unique(speakers[datasets != "TESS"])
    train_speakers, temporary = train_test_split(
        common_speakers, test_size=0.30, random_state=args.seed
    )
    validation_speakers, test_speakers = train_test_split(
        temporary, test_size=0.50, random_state=args.seed
    )

    # TESS has only two speakers. Keep OAF exclusively in training and YAF
    # exclusively in testing so surprise remains measurable without leakage.
    tess_train_speaker, tess_test_speaker = "TESS:OAF", "TESS:YAF"
    train_mask = np.isin(speakers, train_speakers) | (speakers == tess_train_speaker)
    validation_mask = np.isin(speakers, validation_speakers)
    test_mask = np.isin(speakers, test_speakers) | (speakers == tess_test_speaker)

    assert not set(speakers[train_mask]) & set(speakers[validation_mask])
    assert not set(speakers[train_mask]) & set(speakers[test_mask])
    assert not set(speakers[validation_mask]) & set(speakers[test_mask])

    best_model, best_c, best_f1 = None, None, -1.0
    for c_value in (1.0, 3.0, 10.0):
        candidate = make_model(c_value, args.seed)
        candidate.fit(X_model[train_mask], y[train_mask])
        prediction = candidate.predict(X_model[validation_mask])
        score = f1_score(
            y[validation_mask], prediction, labels=COMMON_LABELS,
            average="macro", zero_division=0,
        )
        print(f"C={c_value:g}，验证集六类 Macro-F1={score:.4f}")
        if score > best_f1:
            best_model, best_c, best_f1 = candidate, c_value, score

    validation_metrics = evaluate(
        y[validation_mask], best_model.predict(X_model[validation_mask])
    )

    final_train_mask = train_mask | validation_mask
    final_model = make_model(best_c, args.seed)
    final_model.fit(X_model[final_train_mask], y[final_train_mask])
    test_prediction = final_model.predict(X_model[test_mask])
    test_metrics = evaluate(y[test_mask], test_prediction)

    metrics = {
        "dataset": "TESS + CREMA-D + EmoDB",
        "model": "speaker-reduced MFCC + spectral statistics + RBF-SVM",
        "split": (
            "speaker-independent; CREMA-D/EmoDB speakers grouped 70/15/15; "
            "TESS OAF=train, YAF=test"
        ),
        "seed": args.seed,
        "best_C": best_c,
        "total_samples": int(len(y)),
        "dataset_samples": {
            name: int(np.sum(datasets == name)) for name in np.unique(datasets)
        },
        "unique_speakers": int(len(np.unique(speakers))),
        "split_speakers": {
            "train": int(len(np.unique(speakers[train_mask]))),
            "validation": int(len(np.unique(speakers[validation_mask]))),
            "test": int(len(np.unique(speakers[test_mask]))),
        },
        "class_distribution": {
            "train": distribution(y[train_mask]),
            "validation": distribution(y[validation_mask]),
            "test": distribution(y[test_mask]),
        },
        "validation": validation_metrics,
        "test": test_metrics,
        "test_by_dataset": {
            name: evaluate(y[test_mask & (datasets == name)], test_prediction[datasets[test_mask] == name])
            for name in np.unique(datasets[test_mask])
        },
    }

    joblib.dump(final_model, args.output_dir / "speech_model.joblib")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    matrix = np.asarray(test_metrics["confusion_matrix"])
    np.savetxt(
        args.output_dir / "confusion_matrix.csv", matrix,
        delimiter=",", fmt="%d", header=",".join(LABELS), comments="",
    )

    summary_keys = (
        "samples", "accuracy", "precision_macro",
        "recall_macro", "f1_macro", "f1_weighted",
    )
    print(json.dumps({
        "datasets": metrics["dataset_samples"],
        "speakers": metrics["split_speakers"],
        "validation": {key: validation_metrics[key] for key in summary_keys},
        "test": {key: test_metrics[key] for key in summary_keys},
    }, ensure_ascii=False, indent=2))
    print(f"模型与评估文件已保存：{args.output_dir}")


if __name__ == "__main__":
    main()
