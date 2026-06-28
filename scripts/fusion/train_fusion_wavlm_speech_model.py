from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from emotion_app.audio_features import _read_audio
from emotion_app.domain import EMOTIONS
from scripts.fusion._fusion_common import sha256_file, write_jsonl


LABEL_TO_ID = {label: index for index, label in enumerate(EMOTIONS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


class AudioDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, max_seconds: float, sample_rate: int):
        self.frame = frame.reset_index(drop=True)
        self.max_samples = int(max_seconds * sample_rate)
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        signal = _read_audio(row.audio_path, target_rate=self.sample_rate).astype(np.float32)
        signal = signal[: self.max_samples]
        if signal.size == 0:
            signal = np.zeros(1, dtype=np.float32)
        signal = signal - float(signal.mean())
        signal = signal / (float(signal.std()) + 1e-6)
        return {
            "audio": signal,
            "label": LABEL_TO_ID[str(row.label)],
            "sample_id": str(row.sample_id),
        }


def collate(feature_extractor, sample_rate: int):
    def inner(rows: list[dict]) -> dict:
        audios = [row["audio"] for row in rows]
        encoded = feature_extractor(
            audios,
            sampling_rate=sample_rate,
            padding=True,
            return_tensors="pt",
        )
        return {
            "input_values": encoded["input_values"],
            "attention_mask": encoded.get("attention_mask"),
            "labels": torch.tensor([row["label"] for row in rows], dtype=torch.long),
            "sample_ids": [row["sample_id"] for row in rows],
        }
    return inner


def read_split(data_dir: Path, split: str) -> pd.DataFrame:
    frame = pd.read_csv(data_dir / f"{split}.csv")
    frame = frame[frame["label"].isin(EMOTIONS)].copy()
    frame = frame[frame["audio_path"].map(lambda x: Path(str(x)).is_file())].copy()
    return frame.reset_index(drop=True)


def masked_pool(hidden: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is None:
        mean = hidden.mean(dim=1)
        std = hidden.std(dim=1)
        return torch.cat([mean, std], dim=1)
    # Approximate hidden-frame mask by interpolation from sample mask.
    mask = attention_mask.float().unsqueeze(1)
    mask = torch.nn.functional.interpolate(mask, size=hidden.shape[1], mode="nearest").squeeze(1)
    mask = mask.unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    mean = (hidden * mask).sum(dim=1) / denom
    var = (((hidden - mean.unsqueeze(1)) ** 2) * mask).sum(dim=1) / denom
    std = torch.sqrt(var.clamp_min(1e-8))
    return torch.cat([mean, std], dim=1)


def extract_embeddings(
    frame: pd.DataFrame,
    cache_path: Path,
    model_name: str,
    batch_size: int,
    max_seconds: float,
    sample_rate: int,
    device: torch.device,
    num_workers: int,
    amp: bool,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    labels = frame["label"].map(LABEL_TO_ID).to_numpy(dtype=np.int64)
    sample_ids = frame["sample_id"].astype(str).tolist()
    meta_path = cache_path.with_suffix(".meta.json")
    if cache_path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return np.load(cache_path), labels, meta["sample_ids"]

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    loader = DataLoader(
        AudioDataset(frame, max_seconds=max_seconds, sample_rate=sample_rate),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate(feature_extractor, sample_rate),
        pin_memory=device.type == "cuda",
    )
    embeddings: list[np.ndarray] = []
    ordered_ids: list[str] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"wavlm {cache_path.name}"):
            values = batch["input_values"].to(device)
            mask = batch["attention_mask"]
            if mask is not None:
                mask = mask.to(device)
            with torch.autocast(device_type="cuda", enabled=amp and device.type == "cuda"):
                output = model(input_values=values, attention_mask=mask)
            pooled = masked_pool(output.last_hidden_state.float(), mask)
            embeddings.append(pooled.cpu().numpy())
            ordered_ids.extend(batch["sample_ids"])
    array = np.vstack(embeddings)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, array)
    meta_path.write_text(json.dumps({"model_name": model_name, "sample_ids": ordered_ids}, ensure_ascii=False, indent=2), encoding="utf-8")
    return array, labels, ordered_ids


def build_classifier(kind: str, c: float):
    if kind == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", C=c, n_jobs=-1),
        )
    if kind == "rbf_svm":
        return make_pipeline(
            StandardScaler(),
            SVC(C=c, gamma="scale", class_weight="balanced", probability=True),
        )
    raise ValueError(f"Unsupported classifier: {kind}")


def evaluate(model, x: np.ndarray, y: np.ndarray, sample_ids: list[str]) -> tuple[dict, list[dict]]:
    probs = model.predict_proba(x)
    full = np.zeros((len(x), len(EMOTIONS)), dtype=np.float64)
    for column, class_id in enumerate(model.classes_ if hasattr(model, "classes_") else model[-1].classes_):
        full[:, int(class_id)] = probs[:, column]
    preds = full.argmax(axis=1)
    metrics = {
        "accuracy": accuracy_score(y, preds),
        "macro_f1": f1_score(y, preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(y, preds, average="weighted", zero_division=0),
        "classification_report": classification_report(
            y,
            preds,
            labels=list(range(len(EMOTIONS))),
            target_names=list(EMOTIONS),
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y, preds, labels=list(range(len(EMOTIONS)))).tolist(),
    }
    rows = []
    for sample_id, label_id, prob in zip(sample_ids, y, full):
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
    parser = argparse.ArgumentParser(description="Train WavLM/wav2vec2 embedding speech classifier for fusion datasets.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="microsoft/wavlm-base-plus")
    parser.add_argument("--classifier", choices=["logreg", "rbf_svm"], default="logreg")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seconds", type=float, default=12.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.refresh_cache:
        for path in args.output_dir.glob("*_wavlm.npy"):
            path.unlink()
        for path in args.output_dir.glob("*_wavlm.meta.json"):
            path.unlink()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = not args.no_amp
    frames = {split: read_split(args.data_dir, split) for split in ("train", "validation", "test")}
    arrays = {}
    labels = {}
    sample_ids = {}
    for split, frame in frames.items():
        x, y, ids = extract_embeddings(
            frame,
            args.output_dir / f"{split}_wavlm.npy",
            args.model_name,
            args.batch_size,
            args.max_seconds,
            args.sample_rate,
            device,
            args.num_workers,
            amp,
        )
        arrays[split] = x
        labels[split] = y
        sample_ids[split] = ids

    model = build_classifier(args.classifier, args.C)
    model.fit(arrays["train"], labels["train"])
    val_metrics, val_rows = evaluate(model, arrays["validation"], labels["validation"], sample_ids["validation"])
    test_metrics, test_rows = evaluate(model, arrays["test"], labels["test"], sample_ids["test"])
    model_path = args.output_dir / "wavlm_speech_model.joblib"
    joblib.dump(model, model_path)
    write_jsonl(args.output_dir / "val_speech_probs.jsonl", val_rows)
    write_jsonl(args.output_dir / "test_speech_probs.jsonl", test_rows)
    metrics = {
        "model_name": args.model_name,
        "classifier": args.classifier,
        "C": args.C,
        "label_order": list(EMOTIONS),
        "validation": val_metrics,
        "test": test_metrics,
        "data": {split: len(frame) for split, frame in frames.items()},
        "sha256": {"wavlm_speech_model.joblib": sha256_file(model_path)},
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"validation_accuracy": val_metrics["accuracy"], "validation_macro_f1": val_metrics["macro_f1"], "test_accuracy": test_metrics["accuracy"], "test_macro_f1": test_metrics["macro_f1"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
