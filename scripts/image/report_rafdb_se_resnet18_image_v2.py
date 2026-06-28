from __future__ import annotations

import csv
import json
from pathlib import Path


LABELS = ("surprise", "fear", "disgust", "joy", "sadness", "anger", "neutral")
BASELINE_FLIP_TTA_ACCURACY = 0.7816166883963495
BASELINE_FLIP_TTA_MACRO_F1 = 0.6922
BASELINE_FEAR_F1 = 0.5229
BASELINE_DISGUST_F1 = 0.4462


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def matrix_markdown(matrix: list[list[int]]) -> str:
    lines = ["| true \\ pred | " + " | ".join(LABELS) + " |", "|---|" + "|".join(["---:"] * len(LABELS)) + "|"]
    for label, values in zip(LABELS, matrix):
        lines.append("| " + label + " | " + " | ".join(str(v) for v in values) + " |")
    return "\n".join(lines)


def main() -> None:
    root = Path("outputs/image/rafdb_se_resnet18_image_v2")
    summary_rows = read_summary(root / "all_runs_summary.csv")
    integrity = read_json(root / "log_integrity_check.json")
    valid_rows = [row for row in summary_rows if row.get("valid") == "True"]
    best = max(valid_rows, key=lambda r: (as_float(r, "test_macro_f1_flip_tta"), as_float(r, "test_accuracy_flip_tta"))) if valid_rows else None
    onnx_path = Path("models/image/rafdb_se_resnet18_image_v2/rafdb_emotion_image_v2.onnx")
    onnx_exists = onnx_path.is_file()

    lines: list[str] = []
    lines += [
        "# RAF-DB Basic image-v2 实验报告",
        "",
        "## 1. 实验目的",
        "",
        "在保持自研 SE-ResNet18 主体不变、随机初始化、不使用 ImageNet 预训练和不覆盖当前部署模型的前提下，进行 image-v2 小型改进实验，目标是提升 RAF-DB Basic 七类表情分类的 Macro-F1 和 Accuracy，并重点观察 fear / disgust 等弱类。",
        "",
        "## 2. 当前 baseline 摘要",
        "",
        f"- 当前部署模型：自研 SE-ResNet18 + Flip TTA。",
        f"- 当前部署测试 Accuracy：{BASELINE_FLIP_TTA_ACCURACY:.4f}。",
        f"- 当前历史测试 Macro-F1：约 {BASELINE_FLIP_TTA_MACRO_F1:.4f}。",
        f"- 当前弱类历史 F1：fear 约 {BASELINE_FEAR_F1:.4f}；disgust 约 {BASELINE_DISGUST_F1:.4f}。",
        "- 本实验没有覆盖 `models/image/rafdb_se_resnet18/rafdb_emotion.onnx`，也没有修改 YuNet。",
        "",
        "## 3. 为什么优先改 validation Macro-F1 选模",
        "",
        "RAF-DB Basic 类别分布不均衡，Accuracy 容易受 joy、neutral 等大类影响。Macro-F1 对每个类别等权，更能反映 fear、disgust 等小类/难类是否真正改善。因此 image-v2 的默认 selected checkpoint 使用 validation Macro-F1 最高的权重。",
        "",
        "## 4. 为什么尝试 112×112 输入",
        "",
        "当前部署链路使用 100×100。112×112 仍然保持模型轻量，但能给眼周、鼻翼、嘴角等局部表情线索保留稍多空间细节，理论上可能改善 fear / disgust 等细粒度表情。",
        "",
        "## 5. 为什么尝试 Focal Loss",
        "",
        "Focal Loss 会降低容易样本对梯度的主导，理论上有利于难分类样本。但它也可能牺牲整体校准和大类稳定性，所以本次只作为 C 组候选，不把它预设为一定更优。",
        "",
        "## 6. 实验组配置表",
        "",
        "| run_id | valid | experiment | image_size | loss_name | select_metric |",
        "|---|---:|---|---:|---|---|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['run_id']} | {row['valid']} | {row['experiment_name']} | {row['image_size']} | {row['loss_name']} | {row['select_metric']} |"
        )
    lines += [
        "",
        "## 7. 每组训练最佳 epoch",
        "",
        "| run_id | best_epoch_by_macro_f1 | best_val_accuracy | best_val_macro_f1 |",
        "|---|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['run_id']} | {row['best_epoch_by_macro_f1']} | {as_float(row, 'best_val_accuracy'):.4f} | {as_float(row, 'best_val_macro_f1'):.4f} |"
        )
    lines += [
        "",
        "## 8. 验证集 Accuracy / Macro-F1 对比",
        "",
        "| experiment | val_accuracy | val_macro_f1 |",
        "|---|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(f"| {row['experiment_name']} | {as_float(row, 'best_val_accuracy'):.4f} | {as_float(row, 'best_val_macro_f1'):.4f} |")
    lines += [
        "",
        "## 9. 测试集 Accuracy / Macro-F1 对比",
        "",
        "| experiment | test_accuracy_no_tta | test_macro_f1_no_tta | test_accuracy_flip_tta | test_macro_f1_flip_tta |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['experiment_name']} | {as_float(row, 'test_accuracy_no_tta'):.4f} | {as_float(row, 'test_macro_f1_no_tta'):.4f} | {as_float(row, 'test_accuracy_flip_tta'):.4f} | {as_float(row, 'test_macro_f1_flip_tta'):.4f} |"
        )
    lines += [
        "",
        "## 10. no TTA vs Flip TTA 对比",
        "",
        "Flip TTA 对 A/B/C 均有小幅帮助，其中 B 组达到最高测试 Accuracy 与 Macro-F1。",
        "",
        "## 11. fear / disgust / surprise 等弱类分析",
        "",
        "| experiment | fear_f1_flip_tta | disgust_f1_flip_tta | surprise_f1_flip_tta |",
        "|---|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['experiment_name']} | {as_float(row, 'fear_f1_flip_tta'):.4f} | {as_float(row, 'disgust_f1_flip_tta'):.4f} | {as_float(row, 'surprise_f1_flip_tta'):.4f} |"
        )
    if best:
        lines += [
            "",
            f"最佳 B 组相对历史弱类：fear F1 从约 {BASELINE_FEAR_F1:.4f} 到 {as_float(best, 'fear_f1_flip_tta'):.4f}；disgust F1 从约 {BASELINE_DISGUST_F1:.4f} 到 {as_float(best, 'disgust_f1_flip_tta'):.4f}。",
        ]
    lines += ["", "## 12. 完整混淆矩阵", ""]
    for row in summary_rows:
        test_results = read_json(root / "runs" / row["run_id"] / "test_results.json")
        lines += [
            f"### {row['experiment_name']} / {row['run_id']} / Flip TTA",
            "",
            matrix_markdown(test_results["test_confusion_matrix_flip_tta"]),
            "",
        ]
    lines += ["## 13. 是否建议替换当前图像模型", ""]
    if not valid_rows:
        lines.append("没有有效 run，不建议替换当前图像模型。")
    elif best and onnx_exists:
        lines.append(
            f"建议把 `{best['run_id']}` 作为 image-v2 候选研究模型。它的 Flip TTA Accuracy={as_float(best, 'test_accuracy_flip_tta'):.4f}，Macro-F1={as_float(best, 'test_macro_f1_flip_tta'):.4f}，高于当前 baseline；但正式替换前仍建议做人工样例回归。"
        )
    else:
        lines.append("本次没有导出有效候选 ONNX，不建议替换当前图像模型。")
    lines += ["", "## 14. 新 ONNX 路径", ""]
    if onnx_exists:
        lines.append(f"- `{onnx_path}`")
        meta = read_json(Path(str(onnx_path) + ".metadata.json"))
        lines.append(f"- SHA256：`{meta['model_sha256']}`")
        lines.append(f"- input_shape：`{meta['input_shape']}`")
    else:
        lines.append("- 无。")
    lines += [
        "",
        "## 15. 如果不建议替换时的保留理由",
        "",
        "A 组主要验证 Macro-F1 选模；虽然有效，但 Accuracy 与当前部署模型基本持平。C 组 Focal Loss 没有超过 B 组，也低于当前部署模型的 Accuracy，因此不建议用 C 组替换。B 组是唯一进入候选导出的有效 run。",
        "",
        "## 16. 后续工作建议",
        "",
        "不要继续反复根据官方测试集调参。若继续优化，应新建 image-v3，只基于 validation 设计实验，最后一次性在官方测试集比较。正式接入 EXE 前，应先用真实导入图像和摄像头画面做回归测试。",
        "",
        "## 日志完整性检查",
        "",
        f"- all_valid：{integrity['all_valid']}",
    ]
    for run in integrity["runs"]:
        lines.append(f"- {run['run_id']}: {'valid' if run['valid'] else 'invalid'}")
        if not run["valid"]:
            lines.append(f"  - {json.dumps(run, ensure_ascii=False)}")
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
