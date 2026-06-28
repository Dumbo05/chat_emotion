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
REQUIRED_FAILURE_PAIRS = {
    ("fear", "surprise"),
    ("fear", "anger"),
    ("disgust", "neutral"),
    ("neutral", "disgust"),
    ("disgust", "anger"),
    ("sadness", "neutral"),
    ("surprise", "fear"),
}
BASELINE_FLIP_TTA_ACCURACY = 0.7816166883963495
BASELINE_FLIP_TTA_MACRO_F1 = 0.6922


@dataclass(frozen=True)
class Experiment:
    name: str
    image_size: int
    loss_name: str
    select_metric: str = "val_macro_f1"
    focal_gamma: float | None = None


EXPERIMENTS = {
    "A": Experiment("macro_select_100", 100, "weighted_cross_entropy"),
    "B": Experiment("macro_select_112", 112, "weighted_cross_entropy"),
    "C": Experiment("macro_select_112_focal", 112, "class_balanced_focal_loss", focal_gamma=2.0),
}


class RafDbImageV2Dataset(Dataset):
    def __init__(self, samples: list[Sample], image_size: int, augment: bool = False):
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
        tensor = torch.from_numpy(np.transpose(array, (2, 0, 1)).copy())
        return tensor, sample.label, str(sample.path)


class FocalLoss(nn.Module):
    def __init__(self, weight: torch.Tensor, gamma: float, label_smoothing: float = 0.0):
        super().__init__()
        self.register_buffer("weight", weight)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(
            logits, labels, weight=self.weight, reduction="none", label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce).clamp(min=1e-6, max=1.0)
        return ((1.0 - pt) ** self.gamma * ce).mean()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=True,
            timeout=5,
        )
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
                if isinstance(value, (dict, list, tuple)):
                    encoded[key] = json.dumps(json_ready(value), ensure_ascii=False)
                else:
                    encoded[key] = value
            writer.writerow(encoded)


def class_distribution(samples: list[Sample]) -> dict[str, int]:
    counts = Counter(item.label for item in samples)
    return {RAF_LABELS[index]: int(counts.get(index, 0)) for index in range(len(RAF_LABELS))}


def class_weights_from(samples: list[Sample]) -> torch.Tensor:
    counts = Counter(item.label for item in samples)
    return torch.tensor(
        [len(samples) / (len(RAF_LABELS) * counts[index]) for index in range(len(RAF_LABELS))],
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


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    flip_tta: bool = False,
) -> dict[str, Any]:
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
            flip_logits = model(torch.flip(images, dims=[3]))
            logits = (logits + flip_logits) / 2.0
        probs = torch.softmax(logits, dim=1)
        total_loss += float(loss.detach().item()) * labels.size(0)
        total += labels.size(0)
        truth.extend(labels.cpu().tolist())
        predicted.extend(logits.argmax(1).cpu().tolist())
        probabilities.extend(probs.cpu().numpy().tolist())
        paths.extend(list(batch_paths))
    metrics = metric_bundle(truth, predicted)
    metrics.update(
        {
            "loss": total_loss / max(total, 1),
            "truth": truth,
            "predicted": predicted,
            "probabilities": probabilities,
            "paths": paths,
        }
    )
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> dict[str, Any]:
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


def save_checkpoint(path: Path, model: nn.Module, epoch: int, experiment: Experiment, args: argparse.Namespace) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "labels": RAF_LABELS,
            "epoch": epoch,
            "architecture": "se_resnet18",
            "image_size": experiment.image_size,
            "experiment_name": experiment.name,
            "initialized_from": "random",
            "pretrained": False,
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        },
        path,
    )


def measure_latency(model: nn.Module, image_size: int, device: torch.device) -> float:
    model.eval()
    sample = torch.zeros(1, 3, image_size, image_size, device=device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        loops = 50
        for _ in range(loops):
            _ = model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / loops


def build_predictions_rows(
    test_samples: list[Sample],
    no_tta: dict[str, Any],
    flip_tta: dict[str, Any],
) -> list[dict[str, Any]]:
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


def build_failure_rows(prediction_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in prediction_rows:
        pair = (row["true_label"], row["pred_label_flip_tta"])
        if pair in REQUIRED_FAILURE_PAIRS:
            rows.append(
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
    return rows


def validate_no_nan(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(validate_no_nan(v) for v in value.values())
    if isinstance(value, list):
        return all(validate_no_nan(v) for v in value)
    return True


RUN_CONFIG_FIELDS = {
    "run_id",
    "experiment_name",
    "git_commit_hash",
    "created_at",
    "dataset_name",
    "dataset_root",
    "train_sample_count",
    "val_sample_count",
    "test_sample_count",
    "label_order",
    "class_distribution_train",
    "class_distribution_val",
    "class_distribution_test",
    "image_size",
    "model_name",
    "initialized_from",
    "pretrained",
    "loss_name",
    "class_weight_mode",
    "class_weights",
    "label_smoothing",
    "optimizer",
    "learning_rate",
    "weight_decay",
    "batch_size",
    "max_epochs",
    "early_stopping_patience",
    "scheduler",
    "seed",
    "augmentation_config",
    "select_metric",
    "output_dir",
    "checkpoint_dir",
    "notes",
}
EPOCH_FIELDS = [
    "epoch",
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
TEST_REQUIRED_FIELDS = {
    "checkpoint_used",
    "selected_metric",
    "test_accuracy_no_tta",
    "test_macro_f1_no_tta",
    "test_weighted_f1_no_tta",
    "test_per_class_precision_no_tta",
    "test_per_class_recall_no_tta",
    "test_per_class_f1_no_tta",
    "test_confusion_matrix_no_tta",
    "test_accuracy_flip_tta",
    "test_macro_f1_flip_tta",
    "test_weighted_f1_flip_tta",
    "test_per_class_precision_flip_tta",
    "test_per_class_recall_flip_tta",
    "test_per_class_f1_flip_tta",
    "test_confusion_matrix_flip_tta",
    "test_support_per_class",
    "inference_latency_ms_single_image",
    "model_parameter_count",
    "model_file_size_mb",
    "evaluated_at",
}


def check_run_integrity(run_dir: Path, expected_test_count: int, expected_labels: tuple[str, ...]) -> dict[str, Any]:
    required_files = [
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
    result: dict[str, Any] = {"run_id": run_dir.name, "valid": True, "missing_files": [], "errors": []}
    for name in required_files:
        if not (run_dir / name).is_file():
            result["missing_files"].append(name)
    if result["missing_files"]:
        result["valid"] = False
    try:
        config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        missing_config = sorted(RUN_CONFIG_FIELDS - set(config))
        if missing_config:
            result["errors"].append({"run_config_missing_fields": missing_config})
        if tuple(config.get("label_order", [])) != expected_labels:
            result["errors"].append("label_order mismatch")
        if config.get("test_sample_count") != expected_test_count:
            result["errors"].append("test_sample_count mismatch")
        if not validate_no_nan(config):
            result["errors"].append("NaN/null metric in run_config")
    except Exception as exc:
        result["errors"].append(f"run_config read failed: {exc}")
    try:
        epoch_rows = [json.loads(line) for line in (run_dir / "epoch_metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        if not epoch_rows:
            result["errors"].append("epoch_metrics.jsonl empty")
        for row in epoch_rows:
            missing = sorted(set(EPOCH_FIELDS) - set(row))
            if missing:
                result["errors"].append({"epoch_missing_fields": missing, "epoch": row.get("epoch")})
                break
            if not validate_no_nan(row):
                result["errors"].append({"epoch_has_nan_or_null": row.get("epoch")})
                break
    except Exception as exc:
        result["errors"].append(f"epoch_metrics read failed: {exc}")
    try:
        test_results = json.loads((run_dir / "test_results.json").read_text(encoding="utf-8"))
        missing_test = sorted(TEST_REQUIRED_FIELDS - set(test_results))
        if missing_test:
            result["errors"].append({"test_results_missing_fields": missing_test})
        if not validate_no_nan(test_results):
            result["errors"].append("NaN/null metric in test_results")
    except Exception as exc:
        result["errors"].append(f"test_results read failed: {exc}")
    try:
        with (run_dir / "test_predictions.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != expected_test_count:
            result["errors"].append(f"test_predictions row count mismatch: {len(rows)}")
        required_prediction = {
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
        }
        if rows:
            missing = sorted(required_prediction - set(rows[0]))
            if missing:
                result["errors"].append({"test_predictions_missing_fields": missing})
    except Exception as exc:
        result["errors"].append(f"test_predictions read failed: {exc}")
    if result["errors"]:
        result["valid"] = False
    return result


def make_report(output_root: Path, summaries: list[dict[str, Any]], integrity: dict[str, Any], onnx_path: Path | None) -> None:
    lines = [
        "# RAF-DB Basic image-v2 实验报告",
        "",
        "## 1. 实验目的",
        "",
        "在保持自研 SE-ResNet18 主体、随机初始化、RAF-DB Basic 训练设定不变的前提下，优先尝试用 validation Macro-F1 选模、112×112 输入与 Focal Loss 改善七类表情分类，尤其关注 fear / disgust。",
        "",
        "## 2. 当前 baseline 摘要",
        "",
        f"- 当前部署 SE-ResNet18 + Flip TTA：Accuracy {BASELINE_FLIP_TTA_ACCURACY:.4f}；历史 Macro-F1 约 {BASELINE_FLIP_TTA_MACRO_F1:.4f}。",
        "- 当前部署模型不在本实验中覆盖。",
        "",
        "## 3. 方法说明",
        "",
        "- 优先改 validation Macro-F1 选模：RAF-DB 类别不均衡，Accuracy 容易被 joy/neutral 等大类主导，Macro-F1 更适合观察弱类。",
        "- 尝试 112×112 输入：给眼周、鼻唇等局部表情区域保留稍多空间细节。",
        "- 尝试 Focal Loss：降低易分类样本主导程度，期望改善 fear/disgust 等难类。",
        "",
        "## 4. 实验组配置与结果",
        "",
    ]
    valid_summaries = [row for row in summaries if row.get("valid")]
    lines.extend(
        [
            "| run_id | valid | experiment | image_size | loss | best_epoch_macro | val_acc | val_macro_f1 | test_acc_tta | test_macro_f1_tta | fear_f1 | disgust_f1 | recommendation |",
            "|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in summaries:
        lines.append(
            "| {run_id} | {valid} | {experiment_name} | {image_size} | {loss_name} | {best_epoch_by_macro_f1} | {best_val_accuracy:.4f} | {best_val_macro_f1:.4f} | {test_accuracy_flip_tta:.4f} | {test_macro_f1_flip_tta:.4f} | {fear_f1_flip_tta:.4f} | {disgust_f1_flip_tta:.4f} | {recommendation} |".format(
                **row
            )
        )
    lines.extend(["", "## 5. 日志完整性检查", ""])
    for run in integrity.get("runs", []):
        status = "valid" if run.get("valid") else "invalid"
        lines.append(f"- {run.get('run_id')}: {status}")
        if not run.get("valid"):
            lines.append(f"  - 问题：{json.dumps(run, ensure_ascii=False)}")
    lines.extend(["", "## 6. 是否建议替换当前图像模型", ""])
    if not valid_summaries:
        lines.append("没有任何有效 run；不建议替换，也不导出 ONNX。")
    else:
        best = max(valid_summaries, key=lambda row: (row["test_macro_f1_flip_tta"], row["test_accuracy_flip_tta"]))
        if onnx_path:
            lines.append(f"建议将 `{best['run_id']}` 作为 image-v2 候选研究模型；已导出 ONNX：`{onnx_path}`。正式替换仍建议先做人工样例回归。")
        else:
            lines.append("本次有效 run 未达到“明显优于当前模型”的自动导出阈值；不建议替换当前部署模型。")
    lines.extend(
        [
            "",
            "## 7. 后续建议",
            "",
            "不要继续反复根据官方测试集调参；若要进一步优化，应新建 image-v3，仅依据 validation 设计实验，最后一次性在官方测试集比较。",
        ]
    )
    (output_root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_experiment(experiment: Experiment, args: argparse.Namespace, all_samples: list[Sample]) -> dict[str, Any]:
    run_id = f"{experiment.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
    run_dir = args.output_root / "runs" / run_id
    checkpoint_dir = run_dir
    run_dir.mkdir(parents=True, exist_ok=False)

    official_train = [item for item in all_samples if item.split == "train"]
    official_test = [item for item in all_samples if item.split == "test"]
    train_samples, val_samples = train_test_split(
        official_train,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=[item.label for item in official_train],
    )
    class_weights = class_weights_from(train_samples)
    config = {
        "run_id": run_id,
        "experiment_name": experiment.name,
        "git_commit_hash": git_commit_hash(),
        "created_at": utc_now(),
        "dataset_name": "RAF-DB Basic emotion",
        "dataset_root": str(args.data_root),
        "train_sample_count": len(train_samples),
        "val_sample_count": len(val_samples),
        "test_sample_count": len(official_test),
        "label_order": list(RAF_LABELS),
        "class_distribution_train": class_distribution(train_samples),
        "class_distribution_val": class_distribution(val_samples),
        "class_distribution_test": class_distribution(official_test),
        "image_size": experiment.image_size,
        "model_name": "self_built_se_resnet18",
        "initialized_from": "random",
        "pretrained": False,
        "loss_name": experiment.loss_name,
        "class_weight_mode": "inverse_frequency",
        "class_weights": {RAF_LABELS[i]: float(class_weights[i]) for i in range(len(RAF_LABELS))},
        "label_smoothing": args.label_smoothing,
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "early_stopping_patience": args.patience,
        "scheduler": "CosineAnnealingLR",
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
        "select_metric": experiment.select_metric,
        "output_dir": str(run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "notes": "image-v2 independent run; does not overwrite deployed image model",
    }
    if experiment.focal_gamma is not None:
        config["focal_gamma"] = experiment.focal_gamma
    write_json(run_dir / "run_config.json", config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_options = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    train_loader = DataLoader(
        RafDbImageV2Dataset(train_samples, experiment.image_size, augment=True),
        shuffle=True,
        **loader_options,
    )
    eval_loader_options = dict(loader_options)
    val_loader = DataLoader(RafDbImageV2Dataset(val_samples, experiment.image_size), shuffle=False, **eval_loader_options)
    test_loader = DataLoader(RafDbImageV2Dataset(official_test, experiment.image_size), shuffle=False, **eval_loader_options)

    model = SEResNet18().to(device)
    if experiment.loss_name == "class_balanced_focal_loss":
        criterion: nn.Module = FocalLoss(class_weights.to(device), gamma=experiment.focal_gamma or 2.0, label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    best_macro_f1 = -1.0
    best_accuracy = -1.0
    best_epoch_by_macro = 0
    best_epoch_by_accuracy = 0
    best_acc_at_macro = 0.0
    best_macro_at_acc = 0.0
    stale_epochs = 0
    epoch_rows: list[dict[str, Any]] = []
    jsonl_path = run_dir / "epoch_metrics.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train_metrics = train_one_epoch(model, train_loader, criterion, device, optimizer, scaler)
        val_metrics = evaluate_loader(model, val_loader, criterion, device, flip_tta=False)
        scheduler.step()
        is_best_macro = val_metrics["macro_f1"] > best_macro_f1
        is_best_acc = val_metrics["accuracy"] > best_accuracy
        saved_paths: list[str] = []
        if is_best_macro:
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch_by_macro = epoch
            best_acc_at_macro = val_metrics["accuracy"]
            save_checkpoint(run_dir / "best_by_macro_f1.pth", model, epoch, experiment, args)
            saved_paths.append(str(run_dir / "best_by_macro_f1.pth"))
            stale_epochs = 0
        else:
            stale_epochs += 1
        if is_best_acc:
            best_accuracy = val_metrics["accuracy"]
            best_epoch_by_accuracy = epoch
            best_macro_at_acc = val_metrics["macro_f1"]
            save_checkpoint(run_dir / "best_by_accuracy.pth", model, epoch, experiment, args)
            saved_paths.append(str(run_dir / "best_by_accuracy.pth"))
        save_checkpoint(run_dir / "last.pth", model, epoch, experiment, args)
        row = {
            "epoch": epoch,
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
        if stale_epochs >= args.patience:
            print(f"early stopping {run_id} after epoch {epoch}", flush=True)
            break

    write_csv(run_dir / "epoch_metrics.csv", epoch_rows, EPOCH_FIELDS)
    selected_checkpoint = run_dir / "best_by_macro_f1.pth"
    best_summary = {
        "best_epoch_by_macro_f1": best_epoch_by_macro,
        "best_val_macro_f1": best_macro_f1,
        "best_val_accuracy_at_macro_f1_epoch": best_acc_at_macro,
        "best_epoch_by_accuracy": best_epoch_by_accuracy,
        "best_val_accuracy": best_accuracy,
        "best_val_macro_f1_at_accuracy_epoch": best_macro_at_acc,
        "selected_checkpoint": str(selected_checkpoint),
        "selected_metric": experiment.select_metric,
    }
    write_json(run_dir / "best_summary.json", best_summary)

    saved = torch.load(selected_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["state_dict"])
    no_tta = evaluate_loader(model, test_loader, criterion, device, flip_tta=False)
    flip_tta = evaluate_loader(model, test_loader, criterion, device, flip_tta=True)
    latency_ms = measure_latency(model, experiment.image_size, device)
    model_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model_file_size_mb = selected_checkpoint.stat().st_size / (1024 * 1024)
    test_results = {
        "checkpoint_used": str(selected_checkpoint),
        "selected_metric": experiment.select_metric,
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
        "inference_latency_ms_single_image": latency_ms,
        "model_parameter_count": model_parameter_count,
        "model_file_size_mb": model_file_size_mb,
        "evaluated_at": utc_now(),
    }
    write_json(run_dir / "test_results.json", test_results)
    prediction_rows = build_predictions_rows(official_test, no_tta, flip_tta)
    write_csv(
        run_dir / "test_predictions.csv",
        prediction_rows,
        [
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
        ],
    )
    write_csv(run_dir / "failure_cases.csv", build_failure_rows(prediction_rows), [
        "sample_id",
        "image_path_or_relative_path",
        "true_label",
        "pred_label",
        "top1_prob",
        "top2_label",
        "top2_prob",
        "note",
    ])
    matrix_rows = []
    for i, label in enumerate(RAF_LABELS):
        row = {"true_label": label}
        for j, pred_label in enumerate(RAF_LABELS):
            row[pred_label] = test_results["test_confusion_matrix_flip_tta"][i][j]
        matrix_rows.append(row)
    write_csv(run_dir / "confusion_matrix_test.csv", matrix_rows, ["true_label", *RAF_LABELS])

    return {
        "run_id": run_id,
        "experiment_name": experiment.name,
        "image_size": experiment.image_size,
        "loss_name": experiment.loss_name,
        "select_metric": experiment.select_metric,
        "best_epoch_by_macro_f1": best_epoch_by_macro,
        "best_val_accuracy": best_acc_at_macro,
        "best_val_macro_f1": best_macro_f1,
        "test_accuracy_no_tta": test_results["test_accuracy_no_tta"],
        "test_macro_f1_no_tta": test_results["test_macro_f1_no_tta"],
        "test_accuracy_flip_tta": test_results["test_accuracy_flip_tta"],
        "test_macro_f1_flip_tta": test_results["test_macro_f1_flip_tta"],
        "fear_f1_flip_tta": test_results["test_per_class_f1_flip_tta"]["fear"],
        "disgust_f1_flip_tta": test_results["test_per_class_f1_flip_tta"]["disgust"],
        "surprise_f1_flip_tta": test_results["test_per_class_f1_flip_tta"]["surprise"],
        "model_size_mb": model_file_size_mb,
        "recommendation": "pending_integrity_check",
        "selected_checkpoint": str(selected_checkpoint),
    }


def export_onnx_if_better(output_root: Path, model_dir: Path, best: dict[str, Any]) -> Path | None:
    clearly_better = (
        best["test_macro_f1_flip_tta"] >= BASELINE_FLIP_TTA_MACRO_F1 + 0.01
        or best["test_accuracy_flip_tta"] >= BASELINE_FLIP_TTA_ACCURACY + 0.01
    )
    if not clearly_better:
        return None
    run_dir = output_root / "runs" / best["run_id"]
    checkpoint = torch.load(best["selected_checkpoint"], map_location="cpu", weights_only=True)
    model = SEResNet18().eval()
    model.load_state_dict(checkpoint["state_dict"])
    model_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = model_dir / "rafdb_emotion_image_v2.onnx"
    example = torch.zeros(1, 3, int(best["image_size"]), int(best["image_size"]))
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
    export_meta = {
        "onnx_opset": 17,
        "input_shape": [1, 3, int(best["image_size"]), int(best["image_size"])],
        "label_order": list(RAF_LABELS),
        "preprocessing": "RGB; resize to image_size; x / 127.5 - 1",
        "checkpoint_path": best["selected_checkpoint"],
        "model_sha256": digest,
        "exported_at": utc_now(),
    }
    write_json(model_dir / "rafdb_emotion_image_v2.onnx.metadata.json", export_meta)
    shutil.copy2(run_dir / "test_results.json", model_dir / "test_results.json")
    return onnx_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("datasets/project-data/processed/raf-db-basic/aligned"))
    parser.add_argument("--labels", type=Path, default=Path("datasets/project-data/raw/raf-db-basic/extracted/EmoLabel/list_patition_label.txt"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/image/rafdb_se_resnet18_image_v2"))
    parser.add_argument("--model-output-dir", type=Path, default=Path("models/image/rafdb_se_resnet18_image_v2"))
    parser.add_argument("--experiments", nargs="+", default=["A", "B", "C"], choices=sorted(EXPERIMENTS))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eta-min", type=float, default=1e-5)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--no-export", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "runs").mkdir(parents=True, exist_ok=True)
    all_samples = load_samples(args.labels, args.data_root)
    expected_test_count = sum(1 for item in all_samples if item.split == "test")

    summaries: list[dict[str, Any]] = []
    for key in args.experiments:
        summaries.append(run_experiment(EXPERIMENTS[key], args, all_samples))

    integrity_runs = []
    for summary in summaries:
        check = check_run_integrity(args.output_root / "runs" / summary["run_id"], expected_test_count, RAF_LABELS)
        integrity_runs.append(check)
        summary["valid"] = bool(check["valid"])
        if not check["valid"]:
            summary["recommendation"] = "invalid: excluded"
        elif (
            summary["test_macro_f1_flip_tta"] >= BASELINE_FLIP_TTA_MACRO_F1 + 0.01
            or summary["test_accuracy_flip_tta"] >= BASELINE_FLIP_TTA_ACCURACY + 0.01
        ):
            summary["recommendation"] = "valid: candidate for ONNX export"
        else:
            summary["recommendation"] = "valid: keep current deployed model"

    integrity = {
        "checked_at": utc_now(),
        "label_order_expected": list(RAF_LABELS),
        "expected_test_sample_count": expected_test_count,
        "runs": integrity_runs,
        "all_valid": all(item["valid"] for item in integrity_runs),
    }
    write_json(args.output_root / "log_integrity_check.json", integrity)
    summary_fields = [
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
    write_csv(args.output_root / "all_runs_summary.csv", summaries, summary_fields)
    md_lines = ["# all_runs_summary", "", "| " + " | ".join(summary_fields) + " |", "| " + " | ".join(["---"] * len(summary_fields)) + " |"]
    for row in summaries:
        md_lines.append("| " + " | ".join(str(row.get(field, "")) for field in summary_fields) + " |")
    (args.output_root / "all_runs_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    valid_summaries = [row for row in summaries if row.get("valid")]
    onnx_path = None
    if valid_summaries and not args.no_export:
        best = max(valid_summaries, key=lambda row: (row["test_macro_f1_flip_tta"], row["test_accuracy_flip_tta"]))
        onnx_path = export_onnx_if_better(args.output_root, args.model_output_dir, best)
    make_report(args.output_root, summaries, integrity, onnx_path)
    print(json.dumps({"summary": str(args.output_root / "all_runs_summary.csv"), "onnx": str(onnx_path) if onnx_path else None}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
