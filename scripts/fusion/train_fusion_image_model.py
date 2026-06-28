from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from emotion_app.domain import EMOTIONS
from scripts.fusion._fusion_common import sha256_file, write_jsonl


ImageFile.LOAD_TRUNCATED_IMAGES = True
LABEL_TO_ID = {label: index for index, label in enumerate(EMOTIONS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


class FusionImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        path = Path(str(row.image_path))
        image = Image.open(path).convert("RGB")
        return {
            "pixel_values": self.transform(image),
            "label": LABEL_TO_ID[str(row.label)],
            "sample_id": str(row.sample_id),
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(name: str, pretrained: bool) -> nn.Module:
    weights = "DEFAULT" if pretrained else None
    if name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, len(EMOTIONS))
        return model
    if name == "resnet50":
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, len(EMOTIONS))
        return model
    if name == "resnet18":
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, len(EMOTIONS))
        return model
    if name == "convnext_tiny":
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, len(EMOTIONS))
        return model
    raise ValueError(f"Unsupported model: {name}")


def transforms_for(size: int) -> tuple:
    train_tf = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            transforms.RandomAffine(degrees=8, translate=(0.04, 0.04), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return train_tf, eval_tf


def read_split(data_dir: Path, split: str) -> pd.DataFrame:
    frame = pd.read_csv(data_dir / f"{split}.csv")
    frame = frame[frame["label"].isin(EMOTIONS)].copy()
    frame = frame[frame["image_path"].map(lambda x: Path(str(x)).is_file())].copy()
    return frame


def class_weights(frame: pd.DataFrame, power: float, max_weight: float) -> torch.Tensor:
    counts = frame["label"].value_counts().to_dict()
    total = sum(counts.values())
    weights = []
    for label in EMOTIONS:
        count = max(int(counts.get(label, 0)), 1)
        raw = (total / (len(EMOTIONS) * count)) ** power
        weights.append(min(raw, max_weight))
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> tuple[dict, list[dict]]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    rows: list[dict] = []
    with torch.inference_mode():
        for batch in loader:
            pixels = batch["pixel_values"].to(device)
            with torch.autocast(device_type="cuda", enabled=amp and device.type == "cuda"):
                logits = model(pixels)
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            labels = batch["label"].numpy().tolist()
            preds = probs.argmax(axis=1).tolist()
            y_true.extend(labels)
            y_pred.extend(preds)
            for sample_id, label_id, prob in zip(batch["sample_id"], labels, probs):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "label": ID_TO_LABEL[int(label_id)],
                        "image_probs": {emotion: float(prob[index]) for index, emotion in enumerate(EMOTIONS)},
                        "image_ok": True,
                        "image_pred": ID_TO_LABEL[int(prob.argmax())],
                        "image_confidence": float(prob.max()),
                    }
                )
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(range(len(EMOTIONS))),
            target_names=list(EMOTIONS),
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(EMOTIONS)))).tolist(),
    }
    return metrics, rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Train an image classifier from fusion dataset image_path/label CSVs.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", choices=["efficientnet_b0", "resnet18", "resnet50", "convnext_tiny"], default="efficientnet_b0")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--max-class-weight", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = not args.no_amp

    train_frame = read_split(args.data_dir, "train")
    val_frame = read_split(args.data_dir, "validation")
    test_frame = read_split(args.data_dir, "test")
    train_tf, eval_tf = transforms_for(args.image_size)
    train_loader = DataLoader(
        FusionImageDataset(train_frame, train_tf),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        FusionImageDataset(val_frame, eval_tf),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        FusionImageDataset(test_frame, eval_tf),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args.model_name, pretrained=not args.no_pretrained).to(device)
    weights = class_weights(train_frame, args.class_weight_power, args.max_class_weight).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")

    best_score = -1.0
    best_epoch = 0
    history: list[dict] = []
    best_path = args.output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            pixels = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=amp and device.type == "cuda"):
                logits = model(pixels)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        val_metrics, _ = evaluate(model, val_loader, device, amp)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_accuracy": val_metrics["accuracy"],
            "validation_macro_f1": val_metrics["macro_f1"],
            "validation_weighted_f1": val_metrics["weighted_f1"],
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        score = float(val_metrics["macro_f1"])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save({"model_state_dict": model.state_dict(), "args": vars(args), "label_order": list(EMOTIONS)}, best_path)

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_metrics, val_rows = evaluate(model, val_loader, device, amp)
    test_metrics, test_rows = evaluate(model, test_loader, device, amp)
    write_jsonl(args.output_dir / "val_image_probs.jsonl", val_rows)
    write_jsonl(args.output_dir / "test_image_probs.jsonl", test_rows)
    metrics = {
        "model_name": args.model_name,
        "label_order": list(EMOTIONS),
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_score,
        "validation": val_metrics,
        "test": test_metrics,
        "history": history,
        "class_weights": {label: float(weights[index].detach().cpu()) for index, label in enumerate(EMOTIONS)},
        "data": {
            "train": len(train_frame),
            "validation": len(val_frame),
            "test": len(test_frame),
            "data_dir": str(args.data_dir),
        },
        "sha256": {"best_model.pt": sha256_file(best_path)},
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_validation_macro_f1": best_score, "test_accuracy": test_metrics["accuracy"], "test_macro_f1": test_metrics["macro_f1"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
