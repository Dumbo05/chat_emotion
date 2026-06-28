from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_rafdb_model import RAF_LABELS, SEResNet18, Sample, load_samples


APP_PROBABILITY_ORDER = ("anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral")
FER_TO_RAF = {
    "angry": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "joy",
    "sad": "sadness",
    "surprise": "surprise",
    "neutral": "neutral",
}
REQUIRED_FAILURE_PAIRS = {
    ("fear", "surprise"),
    ("fear", "anger"),
    ("disgust", "neutral"),
    ("neutral", "disgust"),
    ("disgust", "anger"),
    ("sadness", "neutral"),
    ("surprise", "fear"),
}
IMAGE_V2_B_ACCURACY = 0.7858539765319427
IMAGE_V2_B_MACRO_F1 = 0.7064881931720992
IMAGE_V2_B_FEAR_F1 = 0.5818181818181818
IMAGE_V2_B_DISGUST_F1 = 0.4773413897280967


@dataclass(frozen=True)
class ImageSample:
    path: Path
    label: int
    split: str
    dataset: str


class EmotionImageDataset(Dataset):
    def __init__(self, samples: list[ImageSample | Sample], image_size: int, augment: bool = False):
        self.samples = samples
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def augment_image(image: np.ndarray) -> np.ndarray:
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
        height, width = image.shape[:2]
        angle = random.uniform(-12.0, 12.0)
        scale = random.uniform(0.92, 1.08)
        tx, ty = random.uniform(-5, 5), random.uniform(-5, 5)
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
        matrix[:, 2] += (tx, ty)
        image = cv2.warpAffine(
            image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
        )
        alpha = random.uniform(0.82, 1.18)
        beta = random.uniform(-18.0, 18.0)
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        if random.random() < 0.15:
            image = cv2.GaussianBlur(image, (3, 3), 0)
        return image

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        sample = self.samples[index]
        image = cv2.imread(str(sample.path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode image: {sample.path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.augment:
            image = self.augment_image(image)
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        array = image.astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(np.transpose(array, (2, 0, 1)).copy()), int(sample.label), str(sample.path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def git_commit_hash() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=True, timeout=5)
        return result.stdout.strip()
    except Exception:
        return None


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
    path.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_ready(payload), ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            encoded = {}
            for key in fieldnames:
                value = row.get(key, "")
                encoded[key] = json.dumps(json_ready(value), ensure_ascii=False) if isinstance(value, (dict, list, tuple)) else value
            writer.writerow(encoded)


def load_fer_samples(root: Path) -> tuple[list[ImageSample], list[ImageSample]]:
    samples_by_split: dict[str, list[ImageSample]] = {"train": [], "test": []}
    label_to_index = {label: i for i, label in enumerate(RAF_LABELS)}
    for split in ("train", "test"):
        split_dir = root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing FER2013 split directory: {split_dir}")
        for fer_name, raf_name in FER_TO_RAF.items():
            class_dir = split_dir / fer_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing FER2013 class directory: {class_dir}")
            for path in sorted(class_dir.glob("*")):
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    samples_by_split[split].append(ImageSample(path, label_to_index[raf_name], split, "FER2013"))
    if not samples_by_split["train"] or not samples_by_split["test"]:
        raise ValueError(f"FER2013 appears empty under {root}")
    return samples_by_split["train"], samples_by_split["test"]


def to_image_samples(samples: list[Sample], dataset: str = "RAF-DB Basic") -> list[ImageSample]:
    return [ImageSample(item.path, item.label, item.split, dataset) for item in samples]


def class_distribution(samples: list[ImageSample | Sample]) -> dict[str, int]:
    counts = Counter(int(item.label) for item in samples)
    return {RAF_LABELS[index]: int(counts.get(index, 0)) for index in range(len(RAF_LABELS))}


def class_weights_from(samples: list[ImageSample | Sample]) -> torch.Tensor:
    counts = Counter(int(item.label) for item in samples)
    return torch.tensor(
        [len(samples) / (len(RAF_LABELS) * max(1, counts[index])) for index in range(len(RAF_LABELS))],
        dtype=torch.float32,
    )


def metric_bundle(truth: list[int], predicted: list[int]) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=list(range(len(RAF_LABELS))), zero_division=0
    )
    macro = precision_recall_fscore_support(truth, predicted, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(truth, predicted, average="weighted", zero_division=0)
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(macro[2]),
        "weighted_f1": float(weighted[2]),
        "per_class_precision": {RAF_LABELS[i]: float(precision[i]) for i in range(len(RAF_LABELS))},
        "per_class_recall": {RAF_LABELS[i]: float(recall[i]) for i in range(len(RAF_LABELS))},
        "per_class_f1": {RAF_LABELS[i]: float(f1[i]) for i in range(len(RAF_LABELS))},
        "support_per_class": {RAF_LABELS[i]: int(support[i]) for i in range(len(RAF_LABELS))},
        "confusion_matrix": confusion_matrix(truth, predicted, labels=list(range(len(RAF_LABELS)))).tolist(),
    }


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, flip_tta: bool = False) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total = 0
    truth: list[int] = []
    predicted: list[int] = []
    probabilities: list[list[float]] = []
    paths: list[str] = []
    for images, labels, batch_paths in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        if flip_tta:
            logits = (logits + model(torch.flip(images, dims=[3]))) / 2.0
        probs = torch.softmax(logits, dim=1)
        total_loss += float(loss.detach().item()) * labels.size(0)
        total += labels.size(0)
        truth.extend(labels.cpu().tolist())
        predicted.extend(logits.argmax(1).cpu().tolist())
        probabilities.extend(probs.cpu().numpy().tolist())
        paths.extend(list(batch_paths))
    metrics = metric_bundle(truth, predicted)
    metrics.update({"loss": total_loss / max(total, 1), "truth": truth, "predicted": predicted, "probabilities": probabilities, "paths": paths})
    return metrics


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler) -> dict[str, Any]:
    model.train()
    total_loss = 0.0
    total = 0
    truth: list[int] = []
    predicted: list[int] = []
    for images, labels, _paths in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach().item()) * labels.size(0)
        total += labels.size(0)
        truth.extend(labels.cpu().tolist())
        predicted.extend(logits.argmax(1).detach().cpu().tolist())
    metrics = metric_bundle(truth, predicted)
    metrics["loss"] = total_loss / max(total, 1)
    return metrics


def save_checkpoint(path: Path, model: nn.Module, epoch: int, stage: str, args: argparse.Namespace) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "labels": RAF_LABELS,
            "epoch": epoch,
            "stage": stage,
            "architecture": "se_resnet18",
            "image_size": args.image_size,
            "initialized_from": "random_then_fer2013_pretrained" if stage == "raf_finetune" else "random",
            "pretrained": False,
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        },
        path,
    )


def prediction_rows(test_samples: list[ImageSample], no_tta: dict[str, Any], flip_tta: dict[str, Any]) -> list[dict[str, Any]]:
    label_to_index = {label: i for i, label in enumerate(RAF_LABELS)}
    rows = []
    for idx, sample in enumerate(test_samples):
        probs = np.asarray(flip_tta["probabilities"][idx], dtype=np.float64)
        order = probs.argsort()[::-1]
        row = {
            "sample_id": Path(no_tta["paths"][idx]).stem,
            "image_path_or_relative_path": str(sample.path),
            "true_label": RAF_LABELS[int(no_tta["truth"][idx])],
            "pred_label_no_tta": RAF_LABELS[int(no_tta["predicted"][idx])],
            "pred_label_flip_tta": RAF_LABELS[int(flip_tta["predicted"][idx])],
            "correct_no_tta": bool(no_tta["predicted"][idx] == no_tta["truth"][idx]),
            "correct_flip_tta": bool(flip_tta["predicted"][idx] == flip_tta["truth"][idx]),
            "top1_prob": float(probs[order[0]]),
            "top2_label": RAF_LABELS[int(order[1])],
            "top2_prob": float(probs[order[1]]),
        }
        for label in APP_PROBABILITY_ORDER:
            row[f"prob_{label}"] = float(probs[label_to_index[label]])
        rows.append(row)
    return rows


def failure_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        pair = (row["true_label"], row["pred_label_flip_tta"])
        if pair in REQUIRED_FAILURE_PAIRS:
            out.append(
                {
                    "sample_id": row["sample_id"],
                    "image_path_or_relative_path": row["image_path_or_relative_path"],
                    "true_label": row["true_label"],
                    "pred_label": row["pred_label_flip_tta"],
                    "top1_prob": row["top1_prob"],
                    "top2_label": row["top2_label"],
                    "top2_prob": row["top2_prob"],
                    "note": f"{pair[0]} -> {pair[1]}",
                }
            )
    return out


EPOCH_FIELDS = [
    "epoch",
    "phase",
    "train_loss",
    "train_accuracy",
    "train_macro_f1",
    "train_per_class_precision",
    "train_per_class_recall",
    "train_per_class_f1",
    "val_loss",
    "val_accuracy",
    "val_macro_f1",
    "val_weighted_f1",
    "val_per_class_precision",
    "val_per_class_recall",
    "val_per_class_f1",
    "val_confusion_matrix",
    "learning_rate",
    "epoch_time_seconds",
    "is_best_by_val_macro_f1",
    "is_best_by_val_accuracy",
    "checkpoint_path_if_saved",
]


def is_finite_tree(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(is_finite_tree(v) for v in value.values())
    if isinstance(value, list):
        return all(is_finite_tree(v) for v in value)
    return True


def check_integrity(run_dir: Path, expected_test_count: int) -> dict[str, Any]:
    required = [
        "run_config.json",
        "epoch_metrics.csv",
        "epoch_metrics.jsonl",
        "best_summary.json",
        "best_by_macro_f1.pth",
        "best_by_accuracy.pth",
        "last.pth",
        "test_results.json",
        "test_predictions.csv",
        "failure_cases.csv",
        "confusion_matrix_test.csv",
    ]
    errors: list[Any] = []
    missing = [name for name in required if not (run_dir / name).is_file()]
    try:
        config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        if tuple(config.get("label_order", [])) != RAF_LABELS:
            errors.append("label_order mismatch")
        if config.get("test_sample_count") != expected_test_count:
            errors.append("test_sample_count mismatch")
        if not is_finite_tree(config):
            errors.append("non-finite value in run_config")
    except Exception as exc:
        errors.append(f"run_config failed: {exc}")
    try:
        rows = [json.loads(line) for line in (run_dir / "epoch_metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            errors.append("epoch_metrics empty")
        for row in rows:
            miss = sorted(set(EPOCH_FIELDS) - set(row))
            if miss:
                errors.append({"epoch_missing_fields": miss, "epoch": row.get("epoch")})
                break
            if not is_finite_tree(row):
                errors.append({"epoch_non_finite": row.get("epoch")})
                break
    except Exception as exc:
        errors.append(f"epoch_metrics failed: {exc}")
    try:
        with (run_dir / "test_predictions.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            prediction_count = sum(1 for _ in csv.DictReader(handle))
        if prediction_count != expected_test_count:
            errors.append(f"test_predictions count mismatch: {prediction_count}")
    except Exception as exc:
        errors.append(f"test_predictions failed: {exc}")
    return {"run_id": run_dir.name, "valid": not missing and not errors, "missing_files": missing, "errors": errors}


def export_onnx(run_dir: Path, model_dir: Path, summary: dict[str, Any]) -> Path | None:
    better = (
        summary["test_macro_f1_flip_tta"] > IMAGE_V2_B_MACRO_F1
        and (summary["fear_f1_flip_tta"] > IMAGE_V2_B_FEAR_F1 or summary["disgust_f1_flip_tta"] > IMAGE_V2_B_DISGUST_F1)
        and summary["test_accuracy_flip_tta"] >= IMAGE_V2_B_ACCURACY - 0.005
    )
    if not better:
        return None
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(summary["selected_checkpoint"], map_location="cpu", weights_only=True)
    model = SEResNet18().eval()
    model.load_state_dict(checkpoint["state_dict"])
    onnx_path = model_dir / "rafdb_emotion_fer_pretrain.onnx"
    example = torch.zeros(1, 3, int(summary["image_size"]), int(summary["image_size"]))
    torch.onnx.export(
        model,
        example,
        onnx_path,
        input_names=["images"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )
    digest = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    write_json(
        model_dir / "rafdb_emotion_fer_pretrain.onnx.metadata.json",
        {
            "onnx_opset": 17,
            "input_shape": [1, 3, int(summary["image_size"]), int(summary["image_size"])],
            "label_order": list(RAF_LABELS),
            "preprocessing": "RGB; resize to image_size; x / 127.5 - 1",
            "checkpoint_path": summary["selected_checkpoint"],
            "model_sha256": digest,
            "exported_at": utc_now(),
        },
    )
    shutil.copy2(run_dir / "test_results.json", model_dir / "test_results.json")
    return onnx_path


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "runs").mkdir(parents=True, exist_ok=True)
    run_id = f"fer_pretrain_raf_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
    run_dir = args.output_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    fer_train, fer_val = load_fer_samples(args.fer_root)
    raf_all = load_samples(args.raf_labels, args.raf_data_root)
    raf_train_all = to_image_samples([item for item in raf_all if item.split == "train"])
    raf_test = to_image_samples([item for item in raf_all if item.split == "test"])
    raf_train, raf_val = train_test_split(
        raf_train_all,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=[item.label for item in raf_train_all],
    )
    if args.debug_sample_limit:
        def limited(samples: list[ImageSample], limit_per_class: int) -> list[ImageSample]:
            buckets: dict[int, list[ImageSample]] = {i: [] for i in range(len(RAF_LABELS))}
            for sample in samples:
                if len(buckets[int(sample.label)]) < limit_per_class:
                    buckets[int(sample.label)].append(sample)
            return [sample for label in range(len(RAF_LABELS)) for sample in buckets[label]]

        fer_train = limited(fer_train, args.debug_sample_limit)
        fer_val = limited(fer_val, max(1, args.debug_sample_limit // 2))
        raf_train = limited(raf_train, args.debug_sample_limit)
        raf_val = limited(raf_val, max(1, args.debug_sample_limit // 2))
        raf_test = limited(raf_test, max(1, args.debug_sample_limit // 2))

    fer_weights = class_weights_from(fer_train)
    raf_weights = class_weights_from(raf_train)
    config = {
        "run_id": run_id,
        "experiment_name": "fer2013_pretrain_rafdb_finetune_112",
        "git_commit_hash": git_commit_hash(),
        "created_at": utc_now(),
        "dataset_name": "FER2013 train/test pretraining + RAF-DB Basic fine-tuning",
        "dataset_root": str(args.raf_data_root),
        "fer_dataset_root": str(args.fer_root),
        "train_sample_count": len(raf_train),
        "val_sample_count": len(raf_val),
        "test_sample_count": len(raf_test),
        "fer_train_sample_count": len(fer_train),
        "fer_val_sample_count": len(fer_val),
        "label_order": list(RAF_LABELS),
        "class_distribution_train": class_distribution(raf_train),
        "class_distribution_val": class_distribution(raf_val),
        "class_distribution_test": class_distribution(raf_test),
        "class_distribution_fer_train": class_distribution(fer_train),
        "class_distribution_fer_val": class_distribution(fer_val),
        "image_size": args.image_size,
        "model_name": "self_built_se_resnet18",
        "initialized_from": "random; FER2013 supervised pretraining before RAF-DB fine-tuning",
        "pretrained": False,
        "loss_name": "weighted_cross_entropy",
        "focal_gamma": None,
        "class_weight_mode": "inverse_frequency_per_stage",
        "class_weights": {RAF_LABELS[i]: float(raf_weights[i]) for i in range(len(RAF_LABELS))},
        "fer_class_weights": {RAF_LABELS[i]: float(fer_weights[i]) for i in range(len(RAF_LABELS))},
        "label_smoothing": args.label_smoothing,
        "optimizer": "AdamW",
        "learning_rate": {"fer_pretrain": args.fer_learning_rate, "raf_finetune": args.raf_learning_rate},
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "max_epochs": {"fer_pretrain": args.fer_epochs, "raf_finetune": args.raf_epochs},
        "early_stopping_patience": args.patience,
        "scheduler": "CosineAnnealingLR per stage",
        "seed": args.seed,
        "augmentation_config": {
            "horizontal_flip": 0.5,
            "rotation_degrees": 12,
            "scale_range": [0.92, 1.08],
            "translation_pixels": 5,
            "brightness_alpha": [0.82, 1.18],
            "brightness_beta": [-18.0, 18.0],
            "gaussian_blur_probability": 0.15,
        },
        "select_metric": "RAF validation Macro-F1 after fine-tuning",
        "output_dir": str(run_dir),
        "checkpoint_dir": str(run_dir),
        "notes": "Independent FER2013 pretraining experiment; does not overwrite image-v2 or deployed models",
    }
    write_json(run_dir / "run_config.json", config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_options = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    fer_train_loader = DataLoader(EmotionImageDataset(fer_train, args.image_size, True), shuffle=True, **loader_options)
    fer_val_loader = DataLoader(EmotionImageDataset(fer_val, args.image_size, False), shuffle=False, **loader_options)
    raf_train_loader = DataLoader(EmotionImageDataset(raf_train, args.image_size, True), shuffle=True, **loader_options)
    raf_val_loader = DataLoader(EmotionImageDataset(raf_val, args.image_size, False), shuffle=False, **loader_options)
    raf_test_loader = DataLoader(EmotionImageDataset(raf_test, args.image_size, False), shuffle=False, **loader_options)

    model = SEResNet18().to(device)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    epoch_rows: list[dict[str, Any]] = []
    jsonl_path = run_dir / "epoch_metrics.jsonl"
    best_macro = -1.0
    best_acc = -1.0
    best_epoch_macro = 0
    best_epoch_acc = 0
    best_acc_at_macro = 0.0
    best_macro_at_acc = 0.0

    global_epoch = 0
    for phase, train_loader, val_loader, weights, epochs, lr in [
        ("fer_pretrain", fer_train_loader, fer_val_loader, fer_weights, args.fer_epochs, args.fer_learning_rate),
        ("raf_finetune", raf_train_loader, raf_val_loader, raf_weights, args.raf_epochs, args.raf_learning_rate),
    ]:
        criterion = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=args.label_smoothing)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=args.eta_min)
        stale = 0
        for local_epoch in range(1, epochs + 1):
            global_epoch += 1
            start = time.perf_counter()
            train_metrics = train_one_epoch(model, train_loader, criterion, device, optimizer, scaler)
            val_metrics = evaluate_loader(model, val_loader, criterion, device, flip_tta=False)
            scheduler.step()
            is_best_macro = phase == "raf_finetune" and val_metrics["macro_f1"] > best_macro
            is_best_acc = phase == "raf_finetune" and val_metrics["accuracy"] > best_acc
            saved_paths: list[str] = []
            if is_best_macro:
                best_macro = val_metrics["macro_f1"]
                best_epoch_macro = global_epoch
                best_acc_at_macro = val_metrics["accuracy"]
                save_checkpoint(run_dir / "best_by_macro_f1.pth", model, global_epoch, phase, args)
                saved_paths.append(str(run_dir / "best_by_macro_f1.pth"))
                stale = 0
            elif phase == "raf_finetune":
                stale += 1
            if is_best_acc:
                best_acc = val_metrics["accuracy"]
                best_epoch_acc = global_epoch
                best_macro_at_acc = val_metrics["macro_f1"]
                save_checkpoint(run_dir / "best_by_accuracy.pth", model, global_epoch, phase, args)
                saved_paths.append(str(run_dir / "best_by_accuracy.pth"))
            save_checkpoint(run_dir / "last.pth", model, global_epoch, phase, args)
            row = {
                "epoch": global_epoch,
                "phase": phase,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "train_per_class_precision": train_metrics["per_class_precision"],
                "train_per_class_recall": train_metrics["per_class_recall"],
                "train_per_class_f1": train_metrics["per_class_f1"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_weighted_f1": val_metrics["weighted_f1"],
                "val_per_class_precision": val_metrics["per_class_precision"],
                "val_per_class_recall": val_metrics["per_class_recall"],
                "val_per_class_f1": val_metrics["per_class_f1"],
                "val_confusion_matrix": val_metrics["confusion_matrix"],
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch_time_seconds": time.perf_counter() - start,
                "is_best_by_val_macro_f1": is_best_macro,
                "is_best_by_val_accuracy": is_best_acc,
                "checkpoint_path_if_saved": ";".join(saved_paths),
            }
            epoch_rows.append(row)
            append_jsonl(jsonl_path, row)
            print(json.dumps({"run_id": run_id, **json_ready(row)}, ensure_ascii=False), flush=True)
            if phase == "raf_finetune" and stale >= args.patience:
                print(f"early stopping RAF fine-tune after local epoch {local_epoch}", flush=True)
                break

    write_csv(run_dir / "epoch_metrics.csv", epoch_rows, EPOCH_FIELDS)
    selected_checkpoint = run_dir / "best_by_macro_f1.pth"
    if not selected_checkpoint.is_file():
        raise RuntimeError("No RAF fine-tune checkpoint was saved; run is invalid")
    best_summary = {
        "best_epoch_by_macro_f1": best_epoch_macro,
        "best_val_macro_f1": best_macro,
        "best_val_accuracy_at_macro_f1_epoch": best_acc_at_macro,
        "best_epoch_by_accuracy": best_epoch_acc,
        "best_val_accuracy": best_acc,
        "best_val_macro_f1_at_accuracy_epoch": best_macro_at_acc,
        "selected_checkpoint": str(selected_checkpoint),
        "selected_metric": "RAF validation Macro-F1",
    }
    write_json(run_dir / "best_summary.json", best_summary)

    saved = torch.load(selected_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["state_dict"])
    criterion = nn.CrossEntropyLoss(weight=raf_weights.to(device), label_smoothing=args.label_smoothing)
    no_tta = evaluate_loader(model, raf_test_loader, criterion, device, flip_tta=False)
    flip_tta = evaluate_loader(model, raf_test_loader, criterion, device, flip_tta=True)
    param_count = sum(p.numel() for p in model.parameters())
    model_size_mb = selected_checkpoint.stat().st_size / (1024 * 1024)
    test_results = {
        "checkpoint_used": str(selected_checkpoint),
        "selected_metric": "RAF validation Macro-F1",
        "test_accuracy_no_tta": no_tta["accuracy"],
        "test_macro_f1_no_tta": no_tta["macro_f1"],
        "test_weighted_f1_no_tta": no_tta["weighted_f1"],
        "test_per_class_precision_no_tta": no_tta["per_class_precision"],
        "test_per_class_recall_no_tta": no_tta["per_class_recall"],
        "test_per_class_f1_no_tta": no_tta["per_class_f1"],
        "test_confusion_matrix_no_tta": no_tta["confusion_matrix"],
        "test_accuracy_flip_tta": flip_tta["accuracy"],
        "test_macro_f1_flip_tta": flip_tta["macro_f1"],
        "test_weighted_f1_flip_tta": flip_tta["weighted_f1"],
        "test_per_class_precision_flip_tta": flip_tta["per_class_precision"],
        "test_per_class_recall_flip_tta": flip_tta["per_class_recall"],
        "test_per_class_f1_flip_tta": flip_tta["per_class_f1"],
        "test_confusion_matrix_flip_tta": flip_tta["confusion_matrix"],
        "test_support_per_class": flip_tta["support_per_class"],
        "inference_latency_ms_single_image": None,
        "model_parameter_count": param_count,
        "model_file_size_mb": model_size_mb,
        "evaluated_at": utc_now(),
    }
    write_json(run_dir / "test_results.json", test_results)
    rows = prediction_rows(raf_test, no_tta, flip_tta)
    pred_fields = [
        "sample_id",
        "image_path_or_relative_path",
        "true_label",
        "pred_label_no_tta",
        "pred_label_flip_tta",
        "correct_no_tta",
        "correct_flip_tta",
        "prob_anger",
        "prob_disgust",
        "prob_fear",
        "prob_joy",
        "prob_sadness",
        "prob_surprise",
        "prob_neutral",
        "top1_prob",
        "top2_label",
        "top2_prob",
    ]
    write_csv(run_dir / "test_predictions.csv", rows, pred_fields)
    write_csv(run_dir / "failure_cases.csv", failure_rows(rows), ["sample_id", "image_path_or_relative_path", "true_label", "pred_label", "top1_prob", "top2_label", "top2_prob", "note"])
    matrix_rows = []
    for i, label in enumerate(RAF_LABELS):
        row = {"true_label": label}
        for j, pred_label in enumerate(RAF_LABELS):
            row[pred_label] = test_results["test_confusion_matrix_flip_tta"][i][j]
        matrix_rows.append(row)
    write_csv(run_dir / "confusion_matrix_test.csv", matrix_rows, ["true_label", *RAF_LABELS])

    summary = {
        "run_id": run_id,
        "experiment_name": "fer2013_pretrain_rafdb_finetune_112",
        "image_size": args.image_size,
        "loss_name": "weighted_cross_entropy",
        "select_metric": "RAF validation Macro-F1",
        "best_epoch_by_macro_f1": best_epoch_macro,
        "best_val_accuracy": best_acc_at_macro,
        "best_val_macro_f1": best_macro,
        "test_accuracy_no_tta": test_results["test_accuracy_no_tta"],
        "test_macro_f1_no_tta": test_results["test_macro_f1_no_tta"],
        "test_accuracy_flip_tta": test_results["test_accuracy_flip_tta"],
        "test_macro_f1_flip_tta": test_results["test_macro_f1_flip_tta"],
        "fear_f1_flip_tta": test_results["test_per_class_f1_flip_tta"]["fear"],
        "disgust_f1_flip_tta": test_results["test_per_class_f1_flip_tta"]["disgust"],
        "surprise_f1_flip_tta": test_results["test_per_class_f1_flip_tta"]["surprise"],
        "model_size_mb": model_size_mb,
        "selected_checkpoint": str(selected_checkpoint),
    }
    check = check_integrity(run_dir, len(raf_test))
    summary["valid"] = bool(check["valid"])
    if not check["valid"]:
        summary["recommendation"] = "invalid: excluded"
    elif (
        summary["test_macro_f1_flip_tta"] > IMAGE_V2_B_MACRO_F1
        and (summary["fear_f1_flip_tta"] > IMAGE_V2_B_FEAR_F1 or summary["disgust_f1_flip_tta"] > IMAGE_V2_B_DISGUST_F1)
        and summary["test_accuracy_flip_tta"] >= IMAGE_V2_B_ACCURACY - 0.005
    ):
        summary["recommendation"] = "valid: candidate better than image-v2 B"
    else:
        summary["recommendation"] = "valid: keep image-v2 B"

    write_json(args.output_root / "log_integrity_check.json", {"checked_at": utc_now(), "expected_test_sample_count": len(raf_test), "label_order_expected": list(RAF_LABELS), "runs": [check], "all_valid": bool(check["valid"])})
    fields = [
        "run_id",
        "experiment_name",
        "image_size",
        "loss_name",
        "select_metric",
        "best_epoch_by_macro_f1",
        "best_val_accuracy",
        "best_val_macro_f1",
        "test_accuracy_no_tta",
        "test_macro_f1_no_tta",
        "test_accuracy_flip_tta",
        "test_macro_f1_flip_tta",
        "fear_f1_flip_tta",
        "disgust_f1_flip_tta",
        "surprise_f1_flip_tta",
        "model_size_mb",
        "recommendation",
        "valid",
    ]
    write_csv(args.output_root / "all_runs_summary.csv", [summary], fields)
    md_lines = ["# FER2013 pretrain + RAF-DB fine-tune summary", "", "| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    md_lines.append("| " + " | ".join(str(summary.get(field, "")) for field in fields) + " |")
    (args.output_root / "all_runs_summary.md").write_text("\n".join(md_lines), encoding="utf-8")
    onnx_path = export_onnx(run_dir, args.model_output_dir, summary) if summary["valid"] and not args.no_export else None
    report_lines = [
        "# FER2013 预训练 + RAF-DB 微调实验报告",
        "",
        "## 实验目的",
        "在不覆盖 image-v2 和当前部署模型的前提下，测试外部 FER2013 数据预训练是否能提升 RAF-DB Basic 上的 SE-ResNet18 表现，特别关注 fear 与 disgust。",
        "",
        "## 关键结果",
        f"- run_id: `{run_id}`",
        f"- valid: {summary['valid']}",
        f"- best RAF validation Macro-F1: {summary['best_val_macro_f1']:.4f}",
        f"- RAF test Accuracy + Flip TTA: {summary['test_accuracy_flip_tta']:.4f}",
        f"- RAF test Macro-F1 + Flip TTA: {summary['test_macro_f1_flip_tta']:.4f}",
        f"- fear F1 + Flip TTA: {summary['fear_f1_flip_tta']:.4f}",
        f"- disgust F1 + Flip TTA: {summary['disgust_f1_flip_tta']:.4f}",
        f"- recommendation: {summary['recommendation']}",
        "",
        "## 与 image-v2 B 的边界比较",
        f"- image-v2 B: Accuracy={IMAGE_V2_B_ACCURACY:.4f}, Macro-F1={IMAGE_V2_B_MACRO_F1:.4f}, fear F1={IMAGE_V2_B_FEAR_F1:.4f}, disgust F1={IMAGE_V2_B_DISGUST_F1:.4f}",
        "- 本实验使用 FER2013 外部数据预训练，因此不能与纯 RAF-DB 随机训练叙事混写。",
        "",
        "## ONNX 导出",
        f"- onnx_path: `{onnx_path}`" if onnx_path else "- 未导出 ONNX：未同时满足明显优于 image-v2 B 的规则，或 run 无效。",
    ]
    (args.output_root / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(json.dumps({"summary": str(args.output_root / "all_runs_summary.csv"), "onnx": str(onnx_path) if onnx_path else None, "valid": summary["valid"]}, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fer-root", type=Path, default=Path("datasets/project-data/raw/fear2013"))
    parser.add_argument("--raf-data-root", type=Path, default=Path("datasets/project-data/processed/raf-db-basic/aligned"))
    parser.add_argument("--raf-labels", type=Path, default=Path("datasets/project-data/raw/raf-db-basic/extracted/EmoLabel/list_patition_label.txt"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/image/rafdb_se_resnet18_fer_pretrain"))
    parser.add_argument("--model-output-dir", type=Path, default=Path("models/image/rafdb_se_resnet18_fer_pretrain"))
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--fer-epochs", type=int, default=20)
    parser.add_argument("--raf-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fer-learning-rate", type=float, default=3e-3)
    parser.add_argument("--raf-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eta-min", type=float, default=1e-5)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--debug-sample-limit", type=int, default=0)
    parser.add_argument("--no-export", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
