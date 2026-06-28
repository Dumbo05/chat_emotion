from __future__ import annotations

import argparse
import csv
import json
import random
import sys
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

try:
    import timm
except ImportError as exc:
    raise SystemExit("Missing dependency: pip install timm") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_rafdb_model import RAF_LABELS, Sample, load_samples


FER_TO_RAF = {
    "angry": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "joy",
    "sad": "sadness",
    "surprise": "surprise",
    "neutral": "neutral",
}
EXPRESSION_FOLDER_TO_RAF = {
    "angry": "anger",
    "anger": "anger",
    "disgust": "disgust",
    "disgusted": "disgust",
    "fear": "fear",
    "fearful": "fear",
    "happy": "joy",
    "happiness": "joy",
    "joy": "joy",
    "sad": "sadness",
    "sadness": "sadness",
    "surprise": "surprise",
    "surprised": "surprise",
    "neutral": "neutral",
}
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class ImageSample:
    path: Path
    label: int
    split: str
    dataset: str


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


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


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def to_image_samples(samples: list[Sample], dataset: str) -> list[ImageSample]:
    return [ImageSample(item.path, item.label, item.split, dataset) for item in samples]


def load_fer_samples(root: Path) -> tuple[list[ImageSample], list[ImageSample]]:
    label_to_index = {label: i for i, label in enumerate(RAF_LABELS)}
    by_split: dict[str, list[ImageSample]] = {"train": [], "test": []}
    for split in ("train", "test"):
        split_dir = root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing FER2013 split directory: {split_dir}")
        for fer_name, raf_name in FER_TO_RAF.items():
            class_dir = split_dir / fer_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing FER2013 class directory: {class_dir}")
            for path in sorted(class_dir.glob("*")):
                if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    by_split[split].append(ImageSample(path, label_to_index[raf_name], split, "FER2013"))
    if not by_split["train"] or not by_split["test"]:
        raise ValueError(f"FER2013 appears empty under {root}")
    return by_split["train"], by_split["test"]



def load_expression_folder(root: Path, dataset: str) -> tuple[list[ImageSample], list[ImageSample]]:
    label_to_index = {label: i for i, label in enumerate(RAF_LABELS)}
    train_dir = root / "train"
    val_dir = root / "val"
    if not val_dir.is_dir():
        val_dir = root / "validation"
    if not val_dir.is_dir():
        val_dir = root / "test"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(f"Expected train and val/test splits under: {root}")

    def collect(split_dir: Path, split: str) -> list[ImageSample]:
        samples: list[ImageSample] = []
        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            class_key = class_dir.name.lower().replace(" ", "_")
            raf_name = EXPRESSION_FOLDER_TO_RAF.get(class_key)
            if raf_name is None:
                continue
            label = label_to_index[raf_name]
            for path in sorted(class_dir.rglob("*")):
                if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    samples.append(ImageSample(path, label, split, dataset))
        return samples

    train_samples = collect(train_dir, "train")
    val_samples = collect(val_dir, "val")
    if not train_samples or not val_samples:
        raise ValueError(f"No usable seven-class expression samples found under: {root}")
    return train_samples, val_samples

def class_distribution(samples: list[ImageSample]) -> dict[str, int]:
    counts = Counter(item.label for item in samples)
    return {RAF_LABELS[i]: int(counts.get(i, 0)) for i in range(len(RAF_LABELS))}


def class_weights(samples: list[ImageSample]) -> torch.Tensor:
    counts = Counter(item.label for item in samples)
    return torch.tensor(
        [len(samples) / (len(RAF_LABELS) * max(1, counts[i])) for i in range(len(RAF_LABELS))],
        dtype=torch.float32,
    )


class EmotionDataset(Dataset):
    def __init__(self, samples: list[ImageSample], image_size: int, augment: bool):
        self.samples = samples
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        if random.random() < 0.75:
            scale = random.uniform(0.82, 1.0)
            crop_h = max(8, int(height * scale))
            crop_w = max(8, int(width * scale))
            y = random.randint(0, max(0, height - crop_h))
            x = random.randint(0, max(0, width - crop_w))
            image = image[y : y + crop_h, x : x + crop_w]
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
        angle = random.uniform(-15.0, 15.0)
        scale = random.uniform(0.9, 1.1)
        tx = random.uniform(-0.06, 0.06) * image.shape[1]
        ty = random.uniform(-0.06, 0.06) * image.shape[0]
        matrix = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, scale)
        matrix[:, 2] += (tx, ty)
        image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        alpha = random.uniform(0.75, 1.25)
        beta = random.uniform(-24.0, 24.0)
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        if random.random() < 0.12:
            image = cv2.GaussianBlur(image, (3, 3), 0)
        if random.random() < 0.08:
            noise = np.random.normal(0, 5, image.shape).astype(np.float32)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return image

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        sample = self.samples[index]
        image = cv2.imread(str(sample.path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode image: {sample.path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.augment:
            image = self._augment(image)
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
        array = image.astype(np.float32) / 255.0
        array = (array - MEAN) / STD
        tensor = torch.from_numpy(np.transpose(array, (2, 0, 1)).copy())
        return tensor, int(sample.label), str(sample.path)


def create_loader(samples: list[ImageSample], image_size: int, batch_size: int, workers: int, augment: bool, balanced: bool, device: torch.device) -> DataLoader:
    dataset = EmotionDataset(samples, image_size=image_size, augment=augment)
    sampler = None
    shuffle = augment
    if balanced:
        counts = Counter(item.label for item in samples)
        weights = [1.0 / max(1, counts[item.label]) for item in samples]
        sampler = WeightedRandomSampler(weights, num_samples=len(samples), replacement=True)
        shuffle = False
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=workers, pin_memory=device.type == "cuda", persistent_workers=workers > 0)


def metric_bundle(truth: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    predicted = logits.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(truth, predicted, labels=list(range(len(RAF_LABELS))), zero_division=0)
    macro = precision_recall_fscore_support(truth, predicted, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(truth, predicted, average="weighted", zero_division=0)
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(macro[2]),
        "weighted_f1": float(weighted[2]),
        "per_class_f1": {RAF_LABELS[i]: float(f1[i]) for i in range(len(RAF_LABELS))},
        "per_class_precision": {RAF_LABELS[i]: float(precision[i]) for i in range(len(RAF_LABELS))},
        "per_class_recall": {RAF_LABELS[i]: float(recall[i]) for i in range(len(RAF_LABELS))},
        "support_per_class": {RAF_LABELS[i]: int(support[i]) for i in range(len(RAF_LABELS))},
        "confusion_matrix": confusion_matrix(truth, predicted, labels=list(range(len(RAF_LABELS)))).tolist(),
    }


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def train_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, device: torch.device, grad_clip: float) -> dict[str, float]:
    model.train()
    loss_total = 0.0
    correct = 0
    count = 0
    for images, labels, _ in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        loss_total += float(loss.detach().cpu()) * labels.size(0)
        correct += int((logits.argmax(1) == labels).sum().detach().cpu())
        count += labels.size(0)
    return {"loss": loss_total / count, "accuracy": correct / count}


@torch.no_grad()
def collect_logits(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, flip_tta: bool) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    logits_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    paths: list[str] = []
    for images, labels, batch_paths in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        if flip_tta:
            logits = (logits + model(torch.flip(images, dims=[3]))) / 2.0
        loss = criterion(logits, labels)
        losses.extend([float(loss.detach().cpu())] * labels.size(0))
        logits_all.append(logits.detach().cpu().numpy())
        labels_all.append(labels.detach().cpu().numpy())
        paths.extend(batch_paths)
    logits_np = np.concatenate(logits_all, axis=0)
    labels_np = np.concatenate(labels_all, axis=0)
    metrics = metric_bundle(labels_np, logits_np)
    metrics["loss"] = float(np.mean(losses))
    return {"metrics": metrics, "logits": logits_np, "labels": labels_np, "paths": np.asarray(paths)}


def run_stage(stage: str, model: nn.Module, train_samples: list[ImageSample], val_samples: list[ImageSample], args: argparse.Namespace, device: torch.device, run_dir: Path, epochs: int, learning_rate: float, balanced_sampler: bool) -> tuple[nn.Module, dict[str, Any]]:
    train_loader = create_loader(train_samples, args.image_size, args.batch_size, args.workers, True, balanced_sampler, device)
    val_loader = create_loader(val_samples, args.image_size, args.eval_batch_size, args.workers, False, False, device)
    weights = class_weights(train_samples).to(device) if args.class_weights else None
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=args.eta_min)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    best_score = -1.0
    stale = 0
    history: list[dict[str, Any]] = []
    checkpoint = run_dir / f"best_{stage}.pth"
    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, scaler, device, args.grad_clip)
        eval_payload = collect_logits(model, val_loader, criterion, device, flip_tta=args.flip_tta)
        val_metrics = eval_payload["metrics"]
        scheduler.step()
        row = {
            "stage": stage,
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "created_at": now(),
        }
        history.append(row)
        print(json.dumps(json_ready(row), ensure_ascii=False), flush=True)
        score = float(val_metrics["macro_f1"])
        if score > best_score:
            best_score = score
            stale = 0
            torch.save({"state_dict": model.state_dict(), "arch": args.model_arch, "stage": stage, "epoch": epoch, "labels": RAF_LABELS, "image_size": args.image_size, "normalization": {"mean": MEAN.tolist(), "std": STD.tolist()}, "val_metrics": val_metrics, "args": json_ready(vars(args))}, checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"{stage}: early stopping at epoch {epoch}", flush=True)
                break
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["state_dict"])
    return model, {"checkpoint": checkpoint, "best_score": best_score, "best_epoch": saved["epoch"], "history": history}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-arch", default="convnext_base.fb_in22k_ft_in1k")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--expression-pretrain-root", type=Path, action="append", default=[], help="Generic ImageFolder expression dataset root. Can be repeated. Expected train/val/class_name layout.")
    parser.add_argument("--fer-root", type=Path, default=Path("datasets/project-data/raw/fear2013"))
    parser.add_argument("--raf-data-root", type=Path, default=Path("datasets/project-data/processed/raf-db-basic/aligned"))
    parser.add_argument("--raf-labels", type=Path, default=Path("datasets/project-data/raw/raf-db-basic/extracted/EmoLabel/list_patition_label.txt"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/image/rafdb_image_v4_strong"))
    parser.add_argument("--run-name", default="")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--fer-epochs", type=int, default=8)
    parser.add_argument("--raf-epochs", type=int, default=45)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--fer-learning-rate", type=float, default=3e-4)
    parser.add_argument("--expression-pretrain-learning-rate", type=float, default=3e-4)
    parser.add_argument("--raf-learning-rate", type=float, default=8e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--eta-min", type=float, default=1e-6)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--class-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--balanced-sampler", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flip-tta", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug-sample-limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    safe_arch = args.model_arch.replace("/", "_").replace(".", "_")
    run_name = args.run_name or f"{safe_arch}_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.output_root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    all_raf = load_samples(args.raf_labels, args.raf_data_root)
    official_train = to_image_samples([item for item in all_raf if item.split == "train"], "RAF-DB Basic")
    official_test = to_image_samples([item for item in all_raf if item.split == "test"], "RAF-DB Basic")
    raf_train, raf_val = train_test_split(official_train, test_size=args.val_ratio, random_state=args.seed, stratify=[item.label for item in official_train])
    if args.debug_sample_limit:
        raf_train = raf_train[: args.debug_sample_limit]
        raf_val = raf_val[: max(16, args.debug_sample_limit // 5)]
        official_test = official_test[: max(16, args.debug_sample_limit // 5)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(args.model_arch, pretrained=not args.no_pretrained, num_classes=len(RAF_LABELS)).to(device)
    write_json(run_dir / "config.json", {"args": vars(args), "run_name": run_name, "device": str(device), "labels": RAF_LABELS, "created_at": now(), "splits": {"raf_train": len(raf_train), "raf_val": len(raf_val), "raf_official_test": len(official_test)}, "class_distribution": {"raf_train": class_distribution(raf_train), "raf_val": class_distribution(raf_val), "raf_official_test": class_distribution(official_test)}})

    stage_summaries: dict[str, Any] = {}
    if args.fer_epochs > 0:
        if args.expression_pretrain_root:
            pretrain_train: list[ImageSample] = []
            pretrain_val: list[ImageSample] = []
            for root in args.expression_pretrain_root:
                train_part, val_part = load_expression_folder(root, root.name)
                pretrain_train.extend(train_part)
                pretrain_val.extend(val_part)
        else:
            pretrain_train, pretrain_val = load_fer_samples(args.fer_root)
        if args.debug_sample_limit:
            pretrain_train = pretrain_train[: args.debug_sample_limit]
            pretrain_val = pretrain_val[: max(16, args.debug_sample_limit // 5)]
        model, stage_summaries["expression_pretrain"] = run_stage("expression_pretrain", model, pretrain_train, pretrain_val, args, device, run_dir, args.fer_epochs, args.expression_pretrain_learning_rate, args.balanced_sampler)

    model, stage_summaries["raf_finetune"] = run_stage("raf_finetune", model, raf_train, raf_val, args, device, run_dir, args.raf_epochs, args.raf_learning_rate, args.balanced_sampler)

    eval_criterion = nn.CrossEntropyLoss()
    val_loader = create_loader(raf_val, args.image_size, args.eval_batch_size, args.workers, False, False, device)
    test_loader = create_loader(official_test, args.image_size, args.eval_batch_size, args.workers, False, False, device)
    val_payload = collect_logits(model, val_loader, eval_criterion, device, flip_tta=args.flip_tta)
    test_payload = collect_logits(model, test_loader, eval_criterion, device, flip_tta=args.flip_tta)
    np.savez_compressed(run_dir / "raf_val_logits.npz", logits=val_payload["logits"], probabilities=softmax(val_payload["logits"]), labels=val_payload["labels"], paths=val_payload["paths"], label_names=np.asarray(RAF_LABELS))
    np.savez_compressed(run_dir / "raf_test_logits.npz", logits=test_payload["logits"], probabilities=softmax(test_payload["logits"]), labels=test_payload["labels"], paths=test_payload["paths"], label_names=np.asarray(RAF_LABELS))
    summary = {"run_name": run_name, "model_arch": args.model_arch, "seed": args.seed, "pretrained": not args.no_pretrained, "expression_pretrain": args.fer_epochs > 0, "expression_pretrain_roots": args.expression_pretrain_root, "stage_summaries": stage_summaries, "raf_val": val_payload["metrics"], "raf_official_test": test_payload["metrics"], "logit_files": {"raf_val": str(run_dir / "raf_val_logits.npz"), "raf_official_test": str(run_dir / "raf_test_logits.npz")}, "completed_at": now()}
    write_json(run_dir / "summary.json", summary)
    write_csv(args.output_root / f"summary_{run_name}.csv", [{"run_name": run_name, "model_arch": args.model_arch, "seed": args.seed, "val_accuracy": val_payload["metrics"]["accuracy"], "val_macro_f1": val_payload["metrics"]["macro_f1"], "test_accuracy": test_payload["metrics"]["accuracy"], "test_macro_f1": test_payload["metrics"]["macro_f1"], "test_fear_f1": test_payload["metrics"]["per_class_f1"]["fear"], "test_disgust_f1": test_payload["metrics"]["per_class_f1"]["disgust"]}], ["run_name", "model_arch", "seed", "val_accuracy", "val_macro_f1", "test_accuracy", "test_macro_f1", "test_fear_f1", "test_disgust_f1"])
    print(json.dumps(json_ready({"run_dir": run_dir, "raf_official_test": test_payload["metrics"]}), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
