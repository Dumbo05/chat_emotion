from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import timm
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: pip install timm") from exc


LABELS = ("surprise", "fear", "disgust", "joy", "sadness", "anger", "neutral")
BEST_RUNS = (
    "efficientnetv2_m_224_seed42",
    "convnext_large_224_seed42",
    "maxvit_base_224_seed42",
)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def export_checkpoint(run_dir: Path, output_dir: Path, opset: int) -> dict[str, Any]:
    checkpoint_path = run_dir / "best_raf_finetune.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    arch = checkpoint["arch"]
    image_size = int(checkpoint.get("image_size", 224))
    model = timm.create_model(arch, pretrained=False, num_classes=len(LABELS))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    output_path = output_dir / f"{run_dir.name}.onnx"
    example = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            model,
            example,
            output_path,
            input_names=["images"],
            output_names=["logits"],
            dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
    return {
        "run": run_dir.name,
        "arch": arch,
        "image_size": image_size,
        "checkpoint": checkpoint_path,
        "onnx": output_path,
        "labels": list(LABELS),
        "normalization": checkpoint.get(
            "normalization",
            {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deploy-root",
        type=Path,
        default=Path("server-results/image-v4/extracted/image_v4_deploy_best"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models/image/rafdb_v4_ensemble"))
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exports = []
    for run in BEST_RUNS:
        run_dir = args.deploy_root / "runs" / run
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
        print(f"exporting {run}...", flush=True)
        exports.append(export_checkpoint(run_dir, args.output_dir, args.opset))

    metadata = {
        "name": "RAF-DB v4 ONNX ensemble",
        "members": exports,
        "ensemble": "average softmax probabilities from flip-TTA logits",
        "input": {
            "shape": [1, 3, 224, 224],
            "color": "RGB",
            "normalization": "ImageNet mean/std after x / 255",
        },
        "labels": list(LABELS),
        "source_metrics": {
            "validation_selected_test_accuracy": 0.8940677966101694,
            "validation_selected_test_macro_f1": 0.8317297880114808,
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(json_ready(metadata), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(json_ready(metadata), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    sys.exit(main())
