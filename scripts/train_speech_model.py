from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from emotion_app.audio_features import extract_audio_features
from emotion_app.domain import EMOTIONS

FOLDER_LABELS = {
    "angry": "anger", "disgust": "disgust", "fear": "fear",
    "happy": "joy", "neutral": "neutral", "sad": "sadness",
    "pleasant_surprise": "surprise", "pleasant_surprised": "surprise",
}
LABELS = list(EMOTIONS)


def discover(dataset: Path):
    emotion_dirs = []
    for directory in dataset.rglob("*"):
        if not directory.is_dir():
            continue
        folder = directory.name.lower()
        key = folder.split("_", 1)[1] if "_" in folder else folder
        if key in FOLDER_LABELS:
            emotion_dirs.append(directory)
    if not emotion_dirs:
        raise ValueError(f"数据集中没有找到情绪文件夹：{dataset}")
    min_depth = min(len(path.relative_to(dataset).parts) for path in emotion_dirs)
    emotion_dirs = [path for path in emotion_dirs if len(path.relative_to(dataset).parts) == min_depth]
    rows = []
    for directory in sorted(emotion_dirs):
        folder = directory.name.lower()
        key = folder.split("_", 1)[1] if "_" in folder else folder
        label = FOLDER_LABELS[key]
        for path in sorted(directory.glob("*.wav")):
            if path.read_bytes()[:4] != b"RIFF":
                print(f"跳过扩展名为 WAV 的非 RIFF 文件：{path}")
                continue
            parts = path.stem.split("_")
            word = parts[1].lower() if len(parts) >= 3 else path.stem.lower()
            speaker = parts[0].upper() if parts else "unknown"
            speaker = "OAF" if speaker == "OA" else speaker
            rows.append((path, label, word, speaker))
    return rows


def make_model(c_value: float, seed: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(C=c_value, kernel="rbf", probability=True,
                           class_weight="balanced", random_state=seed)),
    ])


def evaluate(y_true, y_pred):
    return {
        "samples": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "classification_report": classification_report(
            y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="训练 TESS 语音情感识别模型")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "archive_sound")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "models" / "speech")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "features.npz"

    rows = discover(args.data_dir)
    if cache_path.is_file() and not args.rebuild_cache:
        cached = np.load(cache_path, allow_pickle=True)
        X, y = cached["X"], cached["y"].astype(str)
        words, speakers = cached["words"].astype(str), cached["speakers"].astype(str)
        speakers = np.where(speakers == "OA", "OAF", speakers)
        if len(y) != len(rows):
            raise ValueError("特征缓存与数据集数量不一致，请增加 --rebuild-cache")
        print(f"复用特征缓存：{cache_path}（{len(y)} 条）")
    else:
        features, labels, words_list, speakers_list = [], [], [], []
        for index, (path, label, word, speaker) in enumerate(rows, 1):
            features.append(extract_audio_features(path))
            labels.append(label)
            words_list.append(word)
            speakers_list.append(speaker)
            if index % 100 == 0 or index == len(rows):
                print(f"提取音频特征：{index}/{len(rows)}", flush=True)
        X, y = np.asarray(features), np.asarray(labels)
        words, speakers = np.asarray(words_list), np.asarray(speakers_list)
        np.savez_compressed(cache_path, X=X, y=y, words=words, speakers=speakers)

    unique_words = np.unique(words)
    train_words, temporary_words = train_test_split(
        unique_words, test_size=.30, random_state=args.seed
    )
    validation_words, test_words = train_test_split(
        temporary_words, test_size=.50, random_state=args.seed
    )
    train_mask = np.isin(words, train_words)
    validation_mask = np.isin(words, validation_words)
    test_mask = np.isin(words, test_words)

    best_model, best_c, best_f1 = None, None, -1.0
    for c_value in (1.0, 3.0, 10.0):

        candidate = make_model(c_value, args.seed)

        candidate.fit(X[train_mask], y[train_mask])
        score = f1_score(y[validation_mask], candidate.predict(X[validation_mask]),
                         average="macro", zero_division=0)
        print(f"C={c_value:g}，验证集 Macro-F1={score:.4f}")
        if score > best_f1:
            best_model, best_c, best_f1 = candidate, c_value, score

    validation_metrics = evaluate(y[validation_mask], best_model.predict(X[validation_mask]))

    # After selecting hyperparameters, refit on train + validation while keeping
    # the word-grouped test set untouched.
    final_train_mask = train_mask | validation_mask
    best_model = make_model(best_c, args.seed)
    best_model.fit(X[final_train_mask], y[final_train_mask])
    test_prediction = best_model.predict(X[test_mask])
    test_metrics = evaluate(y[test_mask], test_prediction)

    # TESS contains only two speakers. Report both leave-one-speaker-out
    # directions to make the generalisation boundary visible.
    speaker_holdout = {}
    for held_out in sorted(np.unique(speakers)):
        held_out_mask = speakers == held_out
        speaker_model = make_model(best_c, args.seed)
        speaker_model.fit(X[~held_out_mask], y[~held_out_mask])
        speaker_holdout[held_out] = evaluate(
            y[held_out_mask], speaker_model.predict(X[held_out_mask])
        )
    speaker_holdout["average"] = {
        key: float(np.mean([speaker_holdout[name][key] for name in np.unique(speakers)]))
        for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted")
    }
    metrics = {
        "dataset": "TESS Toronto emotional speech set",
        "model": "MFCC + spectral statistics + RBF-SVM",
        "split": "word-grouped 70/15/15 (same lexical item cannot cross splits)",
        "seed": args.seed,
        "best_C": best_c,
        "total_samples": int(len(y)),
        "speakers": sorted(np.unique(speakers).tolist()),
        "validation": validation_metrics,
        "test": test_metrics,
        "speaker_holdout": speaker_holdout,
    }
    joblib.dump(best_model, args.output_dir / "speech_model.joblib")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    matrix = np.asarray(test_metrics["confusion_matrix"])
    np.savetxt(args.output_dir / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d",
               header=",".join(LABELS), comments="")

    print(json.dumps({"validation": {k: v for k, v in validation_metrics.items() if not isinstance(v, (dict, list))},
                      "test": {k: v for k, v in test_metrics.items() if not isinstance(v, (dict, list))}},
                     ensure_ascii=False, indent=2))
    print(f"模型与评估文件已保存：{args.output_dir}")


if __name__ == "__main__":
    main()








