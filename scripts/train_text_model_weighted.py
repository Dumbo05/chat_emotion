from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emotion_app.domain import EMOTIONS


LABEL_TO_ID = {label: index for index, label in enumerate(EMOTIONS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


class EmotionDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer, max_length: int):
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        encoded = self.tokenizer(str(row.text), truncation=True, max_length=self.max_length, padding=False)
        encoded["labels"] = LABEL_TO_ID[row.label]
        return encoded


def collate(tokenizer):
    def inner(rows: list[dict]) -> dict:
        labels = torch.tensor([row.pop("labels") for row in rows], dtype=torch.long)
        batch = tokenizer.pad(rows, padding=True, return_tensors="pt")
        batch["labels"] = labels
        return batch

    return inner


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(train_frame: pd.DataFrame, power: float, max_weight: float) -> torch.Tensor:
    counts = Counter(train_frame["label"])
    total = sum(counts.values())
    num_labels = len(EMOTIONS)
    weights = []
    for label in EMOTIONS:
        count = max(int(counts.get(label, 0)), 1)
        balanced = total / (num_labels * count)
        weights.append(min(float(balanced) ** power, max_weight))
    weights = np.asarray(weights, dtype=np.float32)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def forward_loss(model, batch: dict, class_weights: torch.Tensor | None, label_smoothing: float):
    labels = batch.pop("labels")
    outputs = model(**batch)
    loss = F.cross_entropy(
        outputs.logits,
        labels,
        weight=class_weights,
        label_smoothing=label_smoothing,
    )
    return loss, outputs.logits, labels


def predict(model, loader, device, class_weights: torch.Tensor | None, label_smoothing: float) -> tuple[list[int], list[int], float]:
    model.eval()
    predictions: list[int] = []
    labels: list[int] = []
    total_loss = 0.0
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss, logits, y = forward_loss(model, batch, class_weights, label_smoothing)
            total_loss += float(loss.item())
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
            labels.extend(y.cpu().tolist())
    return predictions, labels, total_loss / max(len(loader), 1)


def metrics(predictions: list[int], labels: list[int]) -> dict:
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=list(range(len(EMOTIONS))),
            target_names=list(EMOTIONS),
            output_dict=True,
            zero_division=0,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune a weighted text emotion classifier.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--max-class-weight", type=float, default=4.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    args = parser.parse_args()
    seed_everything(args.seed)

    frames = {split: pd.read_csv(args.data_dir / f"{split}.csv") for split in ("train", "validation", "test")}
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    loaders = {
        split: DataLoader(
            EmotionDataset(frame, tokenizer, args.max_length),
            batch_size=args.batch_size,
            shuffle=split == "train",
            collate_fn=collate(tokenizer),
        )
        for split, frame in frames.items()
    }
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(EMOTIONS),
        label2id=LABEL_TO_ID,
        id2label=ID_TO_LABEL,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    class_weights = compute_class_weights(frames["train"], args.class_weight_power, args.max_class_weight).to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    update_steps = max(1, (len(loaders["train"]) * args.epochs) // args.gradient_accumulation)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(update_steps * 0.1),
        num_training_steps=update_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_f1 = -1.0
    history: list[dict] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for step, batch in enumerate(loaders["train"], start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss, _, _ = forward_loss(model, batch, class_weights, args.label_smoothing)
                loss = loss / args.gradient_accumulation
            scaler.scale(loss).backward()
            running_loss += float(loss.item()) * args.gradient_accumulation
            if step % args.gradient_accumulation == 0 or step == len(loaders["train"]):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        val_pred, val_true, val_loss = predict(model, loaders["validation"], device, class_weights, args.label_smoothing)
        val_metrics = metrics(val_pred, val_true)
        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / max(len(loaders["train"]), 1),
                "validation_loss": val_loss,
                "validation_macro_f1": val_metrics["macro_f1"],
                "validation_accuracy": val_metrics["accuracy"],
            }
        )
        print(json.dumps(history[-1], ensure_ascii=False), flush=True)
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

    best_model = AutoModelForSequenceClassification.from_pretrained(args.output_dir).to(device)
    report: dict = {
        "model_name": args.model_name,
        "seed": args.seed,
        "labels": list(EMOTIONS),
        "class_weights": {label: float(class_weights[LABEL_TO_ID[label]].detach().cpu()) for label in EMOTIONS},
        "class_weight_power": args.class_weight_power,
        "max_class_weight": args.max_class_weight,
        "label_smoothing": args.label_smoothing,
        "best_validation_macro_f1": best_f1,
        "history": history,
    }
    test_frame = frames["test"].reset_index(drop=True)
    test_pred, test_true, test_loss = predict(best_model, loaders["test"], device, class_weights, args.label_smoothing)
    report["test"] = {"loss": test_loss, **metrics(test_pred, test_true)}
    confusion = confusion_matrix(test_true, test_pred, labels=list(range(len(EMOTIONS))))
    pd.DataFrame(confusion, index=EMOTIONS, columns=EMOTIONS).to_csv(args.output_dir / "confusion_matrix.csv")
    for language in sorted(test_frame["language"].unique()):
        indices = test_frame.index[test_frame["language"] == language].tolist()
        report[f"test_{language}"] = metrics(
            [test_pred[index] for index in indices],
            [test_true[index] for index in indices],
        )
    (args.output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Best model and metrics saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
