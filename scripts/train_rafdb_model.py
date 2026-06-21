from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


RAF_LABELS = ("surprise", "fear", "disgust", "joy", "sadness", "anger", "neutral")


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    split: str


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_samples(label_file: Path, image_dir: Path) -> list[Sample]:
    samples: list[Sample] = []
    for line in label_file.read_text(encoding="utf-8").splitlines():
        name, raw_label = line.split()
        stem = Path(name).stem + "_aligned.jpg"
        split = "train" if name.startswith("train_") else "test"
        samples.append(Sample(image_dir / stem, int(raw_label) - 1, split))
    missing = [str(item.path) for item in samples if not item.path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} aligned images; first: {missing[0]}")
    return samples


class RafDbDataset(Dataset):
    def __init__(self, samples: list[Sample], augment: bool = False):
        self.samples = samples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _augment(image: np.ndarray) -> np.ndarray:
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
        height, width = image.shape[:2]
        angle = random.uniform(-12.0, 12.0)
        scale = random.uniform(0.92, 1.08)
        tx, ty = random.uniform(-5, 5), random.uniform(-5, 5)
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
        matrix[:, 2] += (tx, ty)
        image = cv2.warpAffine(
            image, matrix, (width, height), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        alpha = random.uniform(0.82, 1.18)
        beta = random.uniform(-18.0, 18.0)
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        if random.random() < 0.15:
            image = cv2.GaussianBlur(image, (3, 3), 0)
        return image

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        image = cv2.imread(str(sample.path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode image: {sample.path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.augment:
            image = self._augment(image)
        image = cv2.resize(image, (100, 100), interpolation=cv2.INTER_AREA)
        array = image.astype(np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(np.transpose(array, (2, 0, 1)).copy())
        return tensor, sample.label


class ConvBnRelu(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            ConvBnRelu(in_channels, out_channels, stride),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.body(x) + self.skip(x))


class RafEmotionNet(nn.Module):
    """Small residual CNN trained from random initialization on RAF-DB Basic."""

    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.features = nn.Sequential(
            ConvBnRelu(3, 32, 2),
            ResidualBlock(32, 32),
            ResidualBlock(32, 64, 2),
            ResidualBlock(64, 64),
            ResidualBlock(64, 96, 2),
            ResidualBlock(96, 96),
            ResidualBlock(96, 160, 2),
            ResidualBlock(160, 160),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.25), nn.Linear(160, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(self.pool(x))


class SEBasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SqueezeExcitation(out_channels)
        self.skip = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.se(self.bn2(self.conv2(x)))
        return self.activation(x + residual)


class SEResNet18(nn.Module):
    """SE-ResNet18 implemented locally and trained without pretrained weights."""

    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.in_channels = 64
        self.stem = ConvBnRelu(3, 64, 2)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.25), nn.Linear(512, num_classes)
        )
        self._initialize_weights()

    def _make_layer(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [SEBasicBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        layers.extend(SEBasicBlock(out_channels, out_channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.classifier(self.pool(x))


def create_model(architecture: str) -> nn.Module:
    if architecture == "rafemotionnet":
        return RafEmotionNet()
    if architecture == "se_resnet18":
        return SEResNet18()
    raise ValueError(f"Unsupported architecture: {architecture}")

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    loss_total = 0.0
    correct = 0
    count = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            loss_total += loss.detach().item() * labels.size(0)
            correct += int((logits.argmax(1) == labels).sum())
            count += labels.size(0)
    return loss_total / count, correct / count


@torch.no_grad()
def predict_all(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[int], list[int]]:
    model.eval()
    truth: list[int] = []
    predicted: list[int] = []
    for images, labels in tqdm(loader, desc="Official test"):
        logits = model(images.to(device, non_blocking=True))
        truth.extend(labels.tolist())
        predicted.extend(logits.argmax(1).cpu().tolist())
    return truth, predicted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("datasets/project-data/processed/raf-db-basic/aligned"))
    parser.add_argument("--labels", type=Path, default=Path("datasets/project-data/raw/raf-db-basic/extracted/EmoLabel/list_patition_label.txt"))
    parser.add_argument("--output", type=Path, default=Path("models/image/rafdb_emotion"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument(
        "--architecture", choices=("rafemotionnet", "se_resnet18"), default="rafemotionnet"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    all_samples = load_samples(args.labels, args.data_root)
    official_train = [item for item in all_samples if item.split == "train"]
    official_test = [item for item in all_samples if item.split == "test"]
    train_samples, val_samples = train_test_split(
        official_train,
        test_size=0.1,
        random_state=args.seed,
        stratify=[item.label for item in official_train],
    )
    counts = Counter(item.label for item in train_samples)
    class_weights = torch.tensor(
        [len(train_samples) / (len(RAF_LABELS) * counts[index]) for index in range(len(RAF_LABELS))],
        dtype=torch.float32,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    loader_options = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=args.workers > 0,
    )
    train_loader = DataLoader(RafDbDataset(train_samples, True), shuffle=True, **loader_options)
    val_loader = DataLoader(RafDbDataset(val_samples), shuffle=False, **loader_options)
    test_loader = DataLoader(RafDbDataset(official_test), shuffle=False, **loader_options)

    model = create_model(args.architecture).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    history: list[dict[str, float | int]] = []
    best_accuracy = -1.0
    stale_epochs = 0
    checkpoint = args.output / "best.pt"
    print(
        f"device={device}; train={len(train_samples)}; val={len(val_samples)}; "
        f"test={len(official_test)}",
        flush=True,
    )
    print("train class counts:", {RAF_LABELS[k]: v for k, v in sorted(counts.items())}, flush=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            stale_epochs = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "labels": RAF_LABELS,
                    "epoch": epoch,
                    "architecture": args.architecture,
                },
                checkpoint,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early stopping after {epoch} epochs")
                break

    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["state_dict"])
    truth, predicted = predict_all(model, test_loader, device)
    report = classification_report(truth, predicted, target_names=RAF_LABELS, output_dict=True, zero_division=0)
    matrix = confusion_matrix(truth, predicted).tolist()
    metadata = {
        "model": args.architecture,
        "training": "random initialization; no pretrained expression classifier or backbone",
        "dataset": "RAF-DB Basic v1.1",
        "dataset_license": "non-commercial research and educational use only",
        "labels": RAF_LABELS,
        "input": {"shape": [1, 3, 100, 100], "color": "RGB", "normalization": "x / 127.5 - 1"},
        "split": {"train": len(train_samples), "validation": len(val_samples), "official_test": len(official_test)},
        "best_epoch": saved["epoch"],
        "best_validation_accuracy": best_accuracy,
        "official_test_accuracy": report["accuracy"],
        "classification_report": report,
        "confusion_matrix": matrix,
        "history": history,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output / "metrics.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    model = model.cpu().eval()
    example = torch.zeros(1, 3, 100, 100)
    torch.onnx.export(
        model,
        example,
        args.output / "rafdb_emotion.onnx",
        input_names=["images"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )
    print(json.dumps({"official_test_accuracy": report["accuracy"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
