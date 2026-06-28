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
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emotion_app.domain import EMOTIONS


LABEL_TO_ID = {label: index for index, label in enumerate(EMOTIONS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}
LANG_TO_ID = {"en": 0, "zh": 1}

GROUPS = ("negative", "positive", "surprise", "neutral")
GROUP_TO_ID = {label: index for index, label in enumerate(GROUPS)}
EMOTION_TO_GROUP = {
    "anger": "negative",
    "disgust": "negative",
    "fear": "negative",
    "sadness": "negative",
    "joy": "positive",
    "surprise": "surprise",
    "neutral": "neutral",
}


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
        label = str(row.label)
        encoded["labels"] = LABEL_TO_ID[label]
        encoded["group_labels"] = GROUP_TO_ID[EMOTION_TO_GROUP[label]]
        encoded["language_labels"] = LANG_TO_ID.get(str(row.language), 0)
        return encoded


def collate(tokenizer):
    def inner(rows: list[dict]) -> dict:
        labels = torch.tensor([row.pop("labels") for row in rows], dtype=torch.long)
        group_labels = torch.tensor([row.pop("group_labels") for row in rows], dtype=torch.long)
        language_labels = torch.tensor([row.pop("language_labels") for row in rows], dtype=torch.long)
        batch = tokenizer.pad(rows, padding=True, return_tensors="pt")
        batch["labels"] = labels
        batch["group_labels"] = group_labels
        batch["language_labels"] = language_labels
        return batch

    return inner


class MultiTaskEmotionModel(nn.Module):
    def __init__(self, model_name: str, dropout: float):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        hidden_size = int(getattr(self.config, "hidden_size"))
        self.dropout = nn.Dropout(dropout)
        self.emotion_classifier = nn.Linear(hidden_size, len(EMOTIONS))
        self.group_classifier = nn.Linear(hidden_size, len(GROUPS))
        self.language_classifier = nn.Linear(hidden_size, len(LANG_TO_ID))

    def pooled(self, outputs, attention_mask: torch.Tensor) -> torch.Tensor:
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**kwargs)
        pooled = self.dropout(self.pooled(outputs, attention_mask))
        return {
            "emotion_logits": self.emotion_classifier(pooled),
            "group_logits": self.group_classifier(pooled),
            "language_logits": self.language_classifier(pooled),
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(train_frame: pd.DataFrame, power: float, max_weight: float) -> torch.Tensor:
    counts = Counter(train_frame["label"])
    total = sum(counts.values())
    weights = []
    for label in EMOTIONS:
        count = max(int(counts.get(label, 0)), 1)
        balanced = total / (len(EMOTIONS) * count)
        weights.append(min(float(balanced) ** power, max_weight))
    weights = np.asarray(weights, dtype=np.float32)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def multitask_loss(outputs: dict, batch: dict, class_weights, label_smoothing: float, group_weight: float, language_weight: float):
    emotion_loss = F.cross_entropy(
        outputs["emotion_logits"],
        batch["labels"],
        weight=class_weights,
        label_smoothing=label_smoothing,
    )
    group_loss = F.cross_entropy(outputs["group_logits"], batch["group_labels"])
    language_loss = F.cross_entropy(outputs["language_logits"], batch["language_labels"])
    return emotion_loss + group_weight * group_loss + language_weight * language_loss


def predict(model, loader, device, class_weights, label_smoothing: float, group_weight: float, language_weight: float):
    model.eval()
    predictions: list[int] = []
    labels: list[int] = []
    total_loss = 0.0
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                token_type_ids=batch.get("token_type_ids"),
            )
            loss = multitask_loss(outputs, batch, class_weights, label_smoothing, group_weight, language_weight)
            total_loss += float(loss.item())
            predictions.extend(outputs["emotion_logits"].argmax(dim=-1).cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())
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


def save_checkpoint(model: MultiTaskEmotionModel, tokenizer, output_dir: Path, metadata: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(output_dir / "encoder")
    tokenizer.save_pretrained(output_dir)
    torch.save(
        {
            "emotion_classifier": model.emotion_classifier.state_dict(),
            "group_classifier": model.group_classifier.state_dict(),
            "language_classifier": model.language_classifier.state_dict(),
            "metadata": metadata,
        },
        output_dir / "multitask_heads.pt",
    )


def load_checkpoint(model_name: str, output_dir: Path, dropout: float, device) -> MultiTaskEmotionModel:
    model = MultiTaskEmotionModel(str(output_dir / "encoder"), dropout=dropout)
    checkpoint = torch.load(output_dir / "multitask_heads.pt", map_location="cpu")
    model.emotion_classifier.load_state_dict(checkpoint["emotion_classifier"])
    model.group_classifier.load_state_dict(checkpoint["group_classifier"])
    model.language_classifier.load_state_dict(checkpoint["language_classifier"])
    return model.to(device)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune a multi-task text emotion model.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--class-weight-power", type=float, default=0.35)
    parser.add_argument("--max-class-weight", type=float, default=3.0)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--group-loss-weight", type=float, default=0.2)
    parser.add_argument("--language-loss-weight", type=float, default=0.05)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskEmotionModel(args.model_name, dropout=args.dropout).to(device)
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

    metadata = {
        "model_name": args.model_name,
        "labels": list(EMOTIONS),
        "groups": list(GROUPS),
        "class_weights": {label: float(class_weights[LABEL_TO_ID[label]].detach().cpu()) for label in EMOTIONS},
        "args": vars(args) | {"data_dir": str(args.data_dir), "output_dir": str(args.output_dir)},
    }

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for step, batch in enumerate(loaders["train"], start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    token_type_ids=batch.get("token_type_ids"),
                )
                loss = multitask_loss(
                    outputs,
                    batch,
                    class_weights,
                    args.label_smoothing,
                    args.group_loss_weight,
                    args.language_loss_weight,
                )
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

        val_pred, val_true, val_loss = predict(
            model,
            loaders["validation"],
            device,
            class_weights,
            args.label_smoothing,
            args.group_loss_weight,
            args.language_loss_weight,
        )
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
            save_checkpoint(model, tokenizer, args.output_dir, metadata)

    best_model = load_checkpoint(args.model_name, args.output_dir, args.dropout, device)
    report: dict = {
        **metadata,
        "best_validation_macro_f1": best_f1,
        "history": history,
    }
    test_frame = frames["test"].reset_index(drop=True)
    test_pred, test_true, test_loss = predict(
        best_model,
        loaders["test"],
        device,
        class_weights,
        args.label_smoothing,
        args.group_loss_weight,
        args.language_loss_weight,
    )
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
