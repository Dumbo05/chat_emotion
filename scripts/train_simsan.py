from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from emotion_app.audio_features import _read_audio
from emotion_app.domain import EMOTIONS
from emotion_app.simsan import SIMSAN
from scripts.train_speech_model import discover_crema, discover_emodb, discover_tess

SAMPLE_RATE = 16_000
FIXED_SAMPLES = SAMPLE_RATE * 4
N_FFT = 512
FRAME_LENGTH = 400
HOP_LENGTH = 160
N_MELS = 64
SPECTROGRAM_FRAMES = 1 + (FIXED_SAMPLES - FRAME_LENGTH) // HOP_LENGTH
LABELS = list(EMOTIONS)
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mel_filterbank() -> np.ndarray:
    hz_to_mel = lambda hz: 2595.0 * np.log10(1.0 + hz / 700.0)
    mel_to_hz = lambda mel: 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    points = np.linspace(hz_to_mel(20.0), hz_to_mel(SAMPLE_RATE / 2), N_MELS + 2)
    bins = np.floor((N_FFT + 1) * mel_to_hz(points) / SAMPLE_RATE).astype(int)
    bank = np.zeros((N_MELS, N_FFT // 2 + 1), dtype=np.float32)
    for index in range(N_MELS):
        left, center, right = bins[index:index + 3]
        center, right = max(center, left + 1), max(right, center + 1)
        bank[index, left:center] = np.arange(left, center) / (center - left)
        bank[index, center:right] = (right - np.arange(center, right)) / (right - center)
    return bank


MEL_BANK = mel_filterbank()
WINDOW = np.hanning(FRAME_LENGTH).astype(np.float32)


def fixed_waveform(path: Path) -> np.ndarray:
    signal = _read_audio(path)
    if len(signal) < FIXED_SAMPLES:
        signal = np.tile(signal, math.ceil(FIXED_SAMPLES / len(signal)))
    if len(signal) > FIXED_SAMPLES:
        start = (len(signal) - FIXED_SAMPLES) // 2
        signal = signal[start:start + FIXED_SAMPLES]
    return signal[:FIXED_SAMPLES].astype(np.float32)


def log_mel(path: Path) -> np.ndarray:
    signal = fixed_waveform(path)
    count = 1 + (len(signal) - FRAME_LENGTH) // HOP_LENGTH
    indices = (
        np.arange(FRAME_LENGTH)[None, :]
        + HOP_LENGTH * np.arange(count)[:, None]
    )
    frames = signal[indices] * WINDOW[None, :]
    power = np.abs(np.fft.rfft(frames, n=N_FFT, axis=1)).astype(np.float32) ** 2
    mel = power @ MEL_BANK.T
    return np.log(np.maximum(mel, 1e-8)).T.astype(np.float32)


def split_group(values: list[str], seed: int) -> tuple[list[str], list[str], list[str]]:
    train, temporary = train_test_split(
        sorted(values), test_size=0.30, random_state=seed
    )
    validation, test = train_test_split(
        temporary, test_size=0.50, random_state=seed
    )
    return sorted(train), sorted(validation), sorted(test)


def create_or_load_split(samples, path: Path, seed: int) -> dict:
    fingerprint = hashlib.sha256(
        "\n".join(sample.cache_key for sample in samples).encode("utf-8")
    ).hexdigest()
    if path.is_file():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["dataset_fingerprint"] != fingerprint:
            raise ValueError("数据集已变化；为保护锁定测试集，请人工审核后删除旧 split manifest")
        return manifest

    crema = sorted({s.speaker for s in samples if s.dataset == "CREMA-D"})
    emodb = sorted({s.speaker for s in samples if s.dataset == "EmoDB"})
    c_train, c_val, c_test = split_group(crema, seed)
    e_train, e_val, e_test = split_group(emodb, seed + 1)
    manifest = {
        "protocol": "SIMSAN locked speaker-independent split v1",
        "seed": seed,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_fingerprint": fingerprint,
        "train_speakers": sorted(c_train + e_train + ["TESS:OAF"]),
        "validation_speakers": sorted(c_val + e_val + ["TESS:YAF"]),
        # This new final set has never been used by previous MFCC experiments.
        # TESS is omitted because both of its speakers were already observed.
        "final_test_speakers": sorted(c_test + e_test),
        "final_test_policy": (
            "sealed during development; evaluate once only after validation target; "
            "six common emotions because no untouched surprise speaker exists"
        ),
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def prepare_spectrogram_cache(samples, cache_path: Path, metadata_path: Path) -> None:
    fingerprint = hashlib.sha256(
        "\n".join(sample.cache_key for sample in samples).encode("utf-8")
    ).hexdigest()
    if cache_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint") == fingerprint:
            print(f"复用 SIMSAN 频谱缓存：{cache_path}")
            return

    shape = (len(samples), N_MELS, SPECTROGRAM_FRAMES)
    values = np.lib.format.open_memmap(
        cache_path, mode="w+", dtype=np.float16, shape=shape
    )
    for index, sample in enumerate(samples):
        values[index] = log_mel(sample.path).astype(np.float16)
        if (index + 1) % 100 == 0 or index + 1 == len(samples):
            print(f"生成 Log-Mel：{index + 1}/{len(samples)}", flush=True)
    values.flush()
    metadata_path.write_text(json.dumps({
        "fingerprint": fingerprint,
        "shape": shape,
        "dtype": "float16",
        "sample_rate": SAMPLE_RATE,
        "fixed_seconds": 4,
        "n_mels": N_MELS,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


class SpectrogramDataset(Dataset):
    def __init__(
        self, cache_path: Path, indices: np.ndarray, labels: np.ndarray,
        speaker_ids: np.ndarray, augment: str,
    ):
        self.values = np.load(cache_path, mmap_mode="r")
        self.indices = np.asarray(indices)
        self.labels = labels
        self.speaker_ids = speaker_ids
        self.augment = augment != "none"
        self.augmentation = augment

    def __len__(self) -> int:
        return len(self.indices)

    def augment_spectrogram(self, values: torch.Tensor) -> torch.Tensor:
        values = values.clone()
        mel_bins, frames = values.shape

        if self.augmentation == "mild":
            fill = values.mean()
            if random.random() < 0.75:
                width = random.randint(4, 20)
                start = random.randint(0, frames - width)
                values[:, start:start + width] = fill
            if random.random() < 0.60:
                width = random.randint(2, 5)
                start = random.randint(0, mel_bins - width)
                values[start:start + width] = fill
            if random.random() < 0.40:
                values += torch.randn_like(values) * random.uniform(0.005, 0.025)
            return values

        if random.random() < 0.7:
            shift = random.randint(-3, 3)
            values = torch.roll(values, shifts=shift, dims=0)
            if shift > 0:
                values[:shift] = values[shift:shift + 1]
            elif shift < 0:
                values[shift:] = values[shift - 1:shift]

        if random.random() < 0.7:
            factor = random.uniform(0.88, 1.12)
            resized = F.interpolate(
                values[None, None], scale_factor=(1.0, factor),
                mode="bilinear", align_corners=False,
            )[0, 0]
            if resized.shape[1] >= frames:
                start = random.randint(0, resized.shape[1] - frames)
                values = resized[:, start:start + frames]
            else:
                repeats = math.ceil(frames / resized.shape[1])
                values = resized.repeat(1, repeats)[:, :frames]

        fill = values.mean()
        if random.random() < 0.8:
            width = random.randint(4, 32)
            start = random.randint(0, frames - width)
            values[:, start:start + width] = fill
        if random.random() < 0.8:
            width = random.randint(2, 8)
            start = random.randint(0, mel_bins - width)
            values[start:start + width] = fill
        if random.random() < 0.6:
            values += torch.randn_like(values) * random.uniform(0.01, 0.05)
        return values

    def __getitem__(self, item: int):
        index = self.indices[item]
        values = torch.from_numpy(np.array(self.values[index], dtype=np.float32))
        if self.augment:
            values = self.augment_spectrogram(values)
        return (
            values,
            torch.tensor(self.labels[index], dtype=torch.long),
            torch.tensor(self.speaker_ids[index], dtype=torch.long),
        )


def loader_metrics(model, loader, device) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    expected, predicted = [], []
    with torch.inference_mode():
        for values, labels, _ in loader:
            logits, _ = model(values.to(device, non_blocking=True), 0.0)
            expected.extend(labels.numpy().tolist())
            predicted.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    y_true, y_pred = np.asarray(expected), np.asarray(predicted)
    label_ids = list(range(len(LABELS)))
    metrics = {
        "samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(
            y_true, y_pred, labels=label_ids, average="macro", zero_division=0
        )),
        "classification_report": classification_report(
            y_true, y_pred, labels=label_ids, target_names=LABELS,
            output_dict=True, zero_division=0,
        ),
    }
    return metrics, y_true, y_pred


def main() -> None:
    parser = argparse.ArgumentParser(description="训练自研 SIMSAN 跨说话人情感网络")
    parser.add_argument("--tess-dir", type=Path, default=PROJECT_ROOT / "datasets/TESS")
    parser.add_argument("--crema-dir", type=Path, default=PROJECT_ROOT / "datasets/CREMA-D")
    parser.add_argument("--emodb-dir", type=Path, default=PROJECT_ROOT / "datasets/EmoDB")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "models" / "speech")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--final-evaluate", action="store_true")
    parser.add_argument("--run-name", default="v1")
    parser.add_argument("--augmentation", choices=("none", "mild", "strong"), default="strong")
    parser.add_argument("--sampling-power", type=float, default=1.0)
    parser.add_argument("--speaker-loss-weight", type=float, default=0.15)
    parser.add_argument("--grl-max", type=float, default=0.20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = (
        discover_tess(args.tess_dir)
        + discover_crema(args.crema_dir)
        + discover_emodb(args.emodb_dir)
    )
    split_path = args.output_dir / "simsan_split_manifest.json"
    split = create_or_load_split(samples, split_path, args.seed)

    cache_path = args.output_dir / "simsan_logmel.npy"
    prepare_spectrogram_cache(
        samples, cache_path, args.output_dir / "simsan_logmel_metadata.json"
    )

    speakers = np.asarray([sample.speaker for sample in samples])
    labels = np.asarray([LABEL_TO_ID[sample.label] for sample in samples])
    train_mask = np.isin(speakers, split["train_speakers"])
    validation_mask = np.isin(speakers, split["validation_speakers"])
    test_mask = np.isin(speakers, split["final_test_speakers"])
    assert not set(speakers[train_mask]) & set(speakers[validation_mask])
    assert not set(speakers[train_mask]) & set(speakers[test_mask])
    assert not set(speakers[validation_mask]) & set(speakers[test_mask])

    train_speaker_map = {
        speaker: index for index, speaker in enumerate(split["train_speakers"])
    }
    speaker_ids = np.asarray([
        train_speaker_map.get(speaker, -1) for speaker in speakers
    ])
    train_indices = np.flatnonzero(train_mask)
    validation_indices = np.flatnonzero(validation_mask)
    test_indices = np.flatnonzero(test_mask)

    train_counts = Counter(labels[train_indices].tolist())
    sample_weights = np.asarray([
        train_counts[labels[index]] ** (-args.sampling_power) for index in train_indices
    ])
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(train_indices), replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_dataset = SpectrogramDataset(
        cache_path, train_indices, labels, speaker_ids, augment=args.augmentation
    )
    validation_dataset = SpectrogramDataset(
        cache_path, validation_indices, labels, speaker_ids, augment="none"
    )
    test_dataset = SpectrogramDataset(
        cache_path, test_indices, labels, speaker_ids, augment="none"
    )
    workers = 0  # Windows-safe and fast because spectrograms are memory-mapped.
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=workers, pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=workers, pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SIMSAN(
        emotion_classes=len(LABELS),
        speaker_classes=len(split["train_speakers"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=2e-6
    )
    emotion_loss = nn.CrossEntropyLoss(label_smoothing=0.05)
    speaker_loss = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_f1, best_epoch, stale = -1.0, 0, 0
    checkpoint_path = args.output_dir / f"simsan_{args.run_name}.pt"
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, correct, seen = 0.0, 0, 0
        progress = (epoch - 1) / max(args.epochs - 1, 1)
        grl_strength = args.grl_max * (2.0 / (1.0 + math.exp(-10 * progress)) - 1.0)

        for values, emotion_targets, speaker_targets in train_loader:
            values = values.to(device, non_blocking=True)
            emotion_targets = emotion_targets.to(device, non_blocking=True)
            speaker_targets = speaker_targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                emotion_logits, speaker_logits = model(values, grl_strength)
                loss = (
                    emotion_loss(emotion_logits, emotion_targets)
                    + args.speaker_loss_weight * speaker_loss(speaker_logits, speaker_targets)
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach()) * len(values)
            correct += int((emotion_logits.argmax(1) == emotion_targets).sum())
            seen += len(values)

        scheduler.step()
        validation, _, _ = loader_metrics(model, validation_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": running_loss / seen,
            "train_accuracy": correct / seen,
            "validation_accuracy": validation["accuracy"],
            "validation_f1_macro": validation["f1_macro"],
            "grl_strength": grl_strength,
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} loss={row['train_loss']:.4f} "
            f"train_acc={row['train_accuracy']:.4f} "
            f"val_acc={row['validation_accuracy']:.4f} "
            f"val_f1={row['validation_f1_macro']:.4f}",
            flush=True,
        )

        if validation["f1_macro"] > best_f1 + 1e-4:
            best_f1, best_epoch, stale = validation["f1_macro"], epoch, 0
            torch.save({
                "architecture": "SIMSAN-v1",
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "labels": LABELS,
                "speaker_classes": len(split["train_speakers"]),
                "best_epoch": best_epoch,
                "validation": validation,
                "split_manifest_sha256": hashlib.sha256(
                    split_path.read_bytes()
                ).hexdigest(),
            }, checkpoint_path)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"验证集连续 {args.patience} 轮未改善，提前停止。")
                break

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    final_validation, _, _ = loader_metrics(model, validation_loader, device)
    report = {
        "architecture": "SIMSAN-v1",
        "dataset": "TESS + CREMA-D + EmoDB",
        "protocol": split["protocol"],
        "best_epoch": best_epoch,
        "validation": final_validation,
        "history": history,
        "final_test_status": "sealed-not-evaluated",
    }

    if args.final_evaluate:
        final_test, _, _ = loader_metrics(model, test_loader, device)
        report["final_test"] = final_test
        report["final_test_status"] = "evaluated-once"
        report["final_test_evaluated_at_utc"] = datetime.now(timezone.utc).isoformat()
        print(
            f"FINAL_TEST accuracy={final_test['accuracy']:.4f} "
            f"macro_f1={final_test['f1_macro']:.4f}"
        )
    else:
        print("最终测试集保持封存；验证达到目标后使用 --final-evaluate 开启一次。")

    (args.output_dir / f"simsan_{args.run_name}_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"SIMSAN 最佳轮次={best_epoch}，验证准确率={final_validation['accuracy']:.4f}，"
        f"Macro-F1={final_validation['f1_macro']:.4f}"
    )


if __name__ == "__main__":
    main()
