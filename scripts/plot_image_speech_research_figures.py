from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEECH_DIR = ROOT / "models" / "speech"
IMAGE_DIR = ROOT / "models" / "image"
OUT = ROOT / "outputs" / "image_speech_research_figures"
LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]

COLORS = {
    "navy": "#315A7D",
    "blue": "#6FA8C9",
    "light_blue": "#C9DFEA",
    "orange": "#D98E5F",
    "light_orange": "#F1D4BF",
    "gray": "#7B8794",
    "light_gray": "#E8ECF0",
    "dark": "#263238",
    "green": "#4F8A70",
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 8,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.7,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="top", ha="left")


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def normalized(matrix: list[list[int]]) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    return values / np.maximum(values.sum(axis=1, keepdims=True), 1)


def draw_confusion(ax: plt.Axes, matrix: list[list[int]], title: str,
                   show_y: bool = True) -> None:
    values = normalized(matrix)
    image = ax.imshow(values, cmap=mpl.colors.LinearSegmentedColormap.from_list(
        "scientific_blues", ["#F7FAFC", "#A9C9DC", "#315A7D"]), vmin=0, vmax=1)
    for row in range(len(LABELS)):
        for col in range(len(LABELS)):
            value = values[row, col]
            if value >= 0.005:
                ax.text(col, row, f"{value * 100:.0f}", ha="center", va="center",
                        fontsize=5.3, color="white" if value > 0.56 else COLORS["dark"])
    ax.set_xticks(range(len(LABELS)), [x[:3].title() for x in LABELS], rotation=45,
                  ha="right")
    ax.set_yticks(range(len(LABELS)),
                  [x[:3].title() for x in LABELS] if show_y else [])
    ax.set_title(title, pad=5, fontweight="bold")
    ax.set_xlabel("Predicted label")
    if show_y:
        ax.set_ylabel("True label")
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def load_model_profile() -> dict:
    pipeline = joblib.load(SPEECH_DIR / "speech_model.joblib")
    scaler = pipeline.named_steps["scaler"]
    classifier = pipeline.named_steps["classifier"]

    def dnn_profile(path: Path) -> dict:
        net = cv2.dnn.readNet(str(path))
        blobs = []
        for layer_id in range(1, len(net.getLayerNames()) + 1):
            blobs.extend(net.getLayer(layer_id).blobs)
        return {
            "layers": len(net.getLayerNames()),
            "parameters": int(sum(np.prod(blob.shape) for blob in blobs)),
            "size_mb": path.stat().st_size / (1024 ** 2),
        }

    expression = dnn_profile(
        IMAGE_DIR / "facial_expression_recognition_mobilefacenet_2022july.onnx")
    detector = dnn_profile(IMAGE_DIR / "face_detection_yunet_2023mar.onnx")
    speech = {
        "features": int(len(scaler.mean_)),
        "support_vectors": int(classifier.support_vectors_.shape[0]),
        "n_support": classifier.n_support_.astype(int).tolist(),
        "C": float(classifier.C),
        "gamma": float(classifier._gamma),
        "size_mb": (SPEECH_DIR / "speech_model.joblib").stat().st_size / (1024 ** 2),
    }
    return {"expression": expression, "detector": detector, "speech": speech}


def figure_speech_performance(metrics: dict, profile: dict) -> None:
    fig = plt.figure(figsize=(7.2, 4.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], width_ratios=[1.05, 1.35])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    regimes = ["Validation\n(current split)", "Overall test\n(n=2,584)",
               "Dataset-level\nsummary"]
    keys = ["accuracy", "f1_macro"]
    vals = np.array([
        [metrics["validation"][key] for key in keys],
        [metrics["test"][key] for key in keys],
        [metrics["speaker_holdout"]["average"][key] for key in keys],
    ]) * 100
    x = np.arange(len(regimes))
    width = 0.32
    bars1 = ax_a.bar(x - width / 2, vals[:, 0], width, color=COLORS["navy"], label="Accuracy")
    bars2 = ax_a.bar(x + width / 2, vals[:, 1], width, color=COLORS["orange"], label="Macro-F1")
    for bars in (bars1, bars2):
        for bar in bars:
            ax_a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                      f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=6)
    ax_a.set_ylim(0, 112)
    ax_a.set_ylabel("Score (%)")
    ax_a.set_xticks(x, regimes)
    ax_a.legend(loc="upper right", ncol=2, fontsize=6)
    ax_a.set_title("Performance depends strongly on evaluation regime", fontweight="bold")
    ax_a.grid(axis="y", color=COLORS["light_gray"], linewidth=0.6)
    panel_label(ax_a, "a")

    report_names = ["Test", "CREMA-D test", "TESS test"]
    reports = [metrics["test"]["classification_report"],
               metrics["speaker_holdout"]["OAF"]["classification_report"],
               metrics["speaker_holdout"]["YAF"]["classification_report"]]
    y = np.arange(len(LABELS))
    for offset, report, name, color in zip((-0.22, 0, 0.22), reports, report_names,
                                            (COLORS["navy"], COLORS["blue"], COLORS["orange"])):
        f1 = np.array([report[label]["f1-score"] for label in LABELS]) * 100
        ax_b.plot(f1, y + offset, "o-", ms=3.5, lw=1.2, color=color, label=name)
    ax_b.set_yticks(y, [label.title() for label in LABELS])
    ax_b.set_xlim(-3, 103)
    ax_b.set_xlabel("Per-class F1 (%)")
    ax_b.invert_yaxis()
    ax_b.grid(axis="x", color=COLORS["light_gray"], linewidth=0.6)
    ax_b.legend(loc="lower right", fontsize=6)
    ax_b.set_title("Cross-speaker errors are emotion specific", fontweight="bold")
    panel_label(ax_b, "b")

    supports = np.array(profile["speech"]["n_support"])
    support_labels = [str(x) for x in joblib.load(
        SPEECH_DIR / "speech_model.joblib").named_steps["classifier"].classes_]
    order = [support_labels.index(label) for label in LABELS]
    supports = supports[order]
    bars = ax_c.barh(np.arange(len(LABELS)), supports, color=COLORS["light_blue"],
                     edgecolor=COLORS["navy"], linewidth=0.7)
    ax_c.set_yticks(np.arange(len(LABELS)), [label.title() for label in LABELS])
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Support vectors")
    ax_c.set_title("SVM decision support by class", fontweight="bold")
    for bar, value in zip(bars, supports):
        ax_c.text(value + 3, bar.get_y() + bar.get_height() / 2, str(value), va="center", fontsize=6)
    ax_c.set_xlim(0, max(supports) * 1.2)
    ax_c.grid(axis="x", color=COLORS["light_gray"], linewidth=0.6)
    panel_label(ax_c, "c")

    holdouts = ["OAF", "YAF"]
    acc = [metrics["speaker_holdout"][h]["accuracy"] * 100 for h in holdouts]
    f1 = [metrics["speaker_holdout"][h]["f1_macro"] * 100 for h in holdouts]
    positions = np.arange(2)
    ax_d.plot(positions, acc, "o-", color=COLORS["navy"], lw=1.5, ms=5, label="Accuracy")
    ax_d.plot(positions, f1, "s-", color=COLORS["orange"], lw=1.5, ms=5, label="Macro-F1")
    for series in (acc, f1):
        for i, value in enumerate(series):
            ax_d.text(i, value + 1.5, f"{value:.1f}", ha="center", fontsize=6)
    ax_d.set_xticks(positions, [f"CREMA-D\nindependent test", f"TESS\nYAF speaker test"])
    ax_d.set_ylim(0, 60)
    ax_d.set_ylabel("Score (%)")
    ax_d.legend(loc="lower center", ncol=2, fontsize=6)
    ax_d.grid(axis="y", color=COLORS["light_gray"], linewidth=0.6)
    ax_d.set_title("Cross-corpus test comparison", fontweight="bold")
    panel_label(ax_d, "d")
    fig.suptitle("Speech emotion recognition: speaker-independent multi-corpus evaluation",
                 fontsize=10, fontweight="bold")
    save_figure(fig, "figure_1_speech_performance")


def figure_confusion_matrices(metrics: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), constrained_layout=True)
    matrices = [metrics["test"]["confusion_matrix"],
                metrics["speaker_holdout"]["OAF"]["confusion_matrix"],
                metrics["speaker_holdout"]["YAF"]["confusion_matrix"]]
    titles = ["Word-grouped test\nAccuracy 100.0%",
              "OAF speaker holdout\nAccuracy 47.4%",
              "YAF speaker holdout\nAccuracy 48.1%"]
    for idx, (ax, matrix, title) in enumerate(zip(axes, matrices, titles)):
        image = draw_confusion(ax, matrix, title, show_y=(idx == 0))
        panel_label(ax, chr(ord("a") + idx))
    cbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Row-normalized proportion")
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["0", "0.5", "1.0"])
    fig.suptitle("Speech confusion matrices across independent test domains",
                 fontsize=10, fontweight="bold")
    save_figure(fig, "figure_2_speech_confusion_matrices")


def rounded_box(ax: plt.Axes, x: float, y: float, w: float, h: float, text: str,
                color: str) -> None:
    box = mpl.patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.025",
                                     facecolor=color, edgecolor="white", linewidth=1)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.5)


def figure_model_profiles(profile: dict) -> None:
    fig = plt.figure(figsize=(7.2, 4.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0], width_ratios=[1.45, 1.0])
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])

    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    ax_a.axis("off")
    rounded_box(ax_a, 0.02, 0.60, 0.15, 0.22, "Input image", COLORS["light_gray"])
    rounded_box(ax_a, 0.23, 0.60, 0.16, 0.22,
                f"YuNet detector\n{profile['detector']['parameters'] / 1e3:.1f}k parameters",
                COLORS["light_blue"])
    rounded_box(ax_a, 0.45, 0.60, 0.16, 0.22, "5-point alignment\n112 x 112 RGB",
                COLORS["light_gray"])
    rounded_box(ax_a, 0.67, 0.60, 0.18, 0.22,
                f"MobileFaceNet FER\n{profile['expression']['parameters'] / 1e6:.2f}M parameters",
                COLORS["light_orange"])
    rounded_box(ax_a, 0.89, 0.60, 0.09, 0.22, "7-class\nsoftmax", COLORS["light_gray"])
    for start, end in ((0.17, 0.23), (0.39, 0.45), (0.61, 0.67), (0.85, 0.89)):
        ax_a.annotate("", xy=(end, 0.71), xytext=(start, 0.71),
                      arrowprops=dict(arrowstyle="->", lw=1, color=COLORS["gray"]))
    ax_a.text(0.02, 0.34, "Image pipeline", fontweight="bold", fontsize=8)
    ax_a.text(0.23, 0.34, "Local independent accuracy: not available", color=COLORS["orange"],
              fontsize=7, fontweight="bold")
    ax_a.text(0.02, 0.13,
              "No labeled image test set or local confusion matrix is present; upstream weights are used for inference.",
              fontsize=6.5, color=COLORS["dark"])
    ax_a.set_title("Auditable image-recognition pipeline and evidence boundary", fontweight="bold")
    panel_label(ax_a, "a")

    names = ["YuNet\ndetector", "MobileFaceNet\nFER"]
    params = np.array([profile["detector"]["parameters"], profile["expression"]["parameters"]])
    bars = ax_b.bar(names, params / 1e6, color=[COLORS["blue"], COLORS["orange"]], width=0.58)
    for bar, value in zip(bars, params):
        ax_b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04,
                  f"{value:,}", ha="center", fontsize=6.5)
    ax_b.set_ylabel("Parameters (millions)")
    ax_b.set_ylim(0, 1.36)
    ax_b.grid(axis="y", color=COLORS["light_gray"], linewidth=0.6)
    ax_b.set_title("Image-model parameter count", fontweight="bold")
    panel_label(ax_b, "b")

    names = ["YuNet", "MobileFaceNet", "Speech SVM"]
    sizes = [profile["detector"]["size_mb"], profile["expression"]["size_mb"],
             profile["speech"]["size_mb"]]
    bars = ax_c.barh(np.arange(3), sizes, color=[COLORS["blue"], COLORS["orange"], COLORS["navy"]])
    ax_c.set_yticks(np.arange(3), names)
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Serialized size (MiB)")
    ax_c.grid(axis="x", color=COLORS["light_gray"], linewidth=0.6)
    for bar, value in zip(bars, sizes):
        ax_c.text(value + 0.08, bar.get_y() + bar.get_height() / 2, f"{value:.2f}",
                  va="center", fontsize=6.5)
    ax_c.set_xlim(0, max(sizes) * 1.22)
    ax_c.set_title("Deployment footprint", fontweight="bold")
    panel_label(ax_c, "c")
    fig.suptitle("Image and speech model profiles for local deployment",
                 fontsize=10, fontweight="bold")
    save_figure(fig, "figure_3_model_profiles")


def export_source_data(metrics: dict, profile: dict) -> None:
    write_csv(OUT / "source_data_speech_summary.csv",
              ["evaluation", "samples", "accuracy", "macro_f1", "macro_precision", "macro_recall"],
              [["validation", metrics["validation"]["samples"], metrics["validation"]["accuracy"],
                metrics["validation"]["f1_macro"], metrics["validation"]["precision_macro"],
                metrics["validation"]["recall_macro"]],
               ["test", metrics["test"]["samples"], metrics["test"]["accuracy"],
                metrics["test"]["f1_macro"], metrics["test"]["precision_macro"],
                metrics["test"]["recall_macro"]],
               ["CREMA-D_test", metrics["speaker_holdout"]["OAF"]["samples"],
                metrics["speaker_holdout"]["OAF"]["accuracy"],
                metrics["speaker_holdout"]["OAF"]["f1_macro"],
                metrics["speaker_holdout"]["OAF"]["precision_macro"],
                metrics["speaker_holdout"]["OAF"]["recall_macro"]],
               ["TESS_test", metrics["speaker_holdout"]["YAF"]["samples"],
                metrics["speaker_holdout"]["YAF"]["accuracy"],
                metrics["speaker_holdout"]["YAF"]["f1_macro"],
                metrics["speaker_holdout"]["YAF"]["precision_macro"],
                metrics["speaker_holdout"]["YAF"]["recall_macro"]]])

    rows = []
    for evaluation, report in (
        ("test", metrics["test"]["classification_report"]),
        ("CREMA-D_test", metrics["speaker_holdout"]["OAF"]["classification_report"]),
        ("TESS_test", metrics["speaker_holdout"]["YAF"]["classification_report"]),
    ):
        for label in LABELS:
            rows.append([evaluation, label, report[label]["precision"], report[label]["recall"],
                         report[label]["f1-score"], report[label]["support"]])
    write_csv(OUT / "source_data_speech_per_class.csv",
              ["evaluation", "class", "precision", "recall", "f1", "support"], rows)

    write_csv(OUT / "source_data_model_profiles.csv",
              ["component", "model", "parameters_or_support_vectors", "layers_or_features",
               "serialized_size_mib", "local_accuracy_available"],
              [["image_detector", "YuNet", profile["detector"]["parameters"],
                profile["detector"]["layers"], profile["detector"]["size_mb"], "no"],
               ["image_classifier", "Progressive Teacher / MobileFaceNet",
                profile["expression"]["parameters"], profile["expression"]["layers"],
                profile["expression"]["size_mb"], "no"],
               ["speech_classifier", "RBF-SVM", profile["speech"]["support_vectors"],
                profile["speech"]["features"], profile["speech"]["size_mb"], "yes"]])

    for name, matrix in (
        ("test", metrics["test"]["confusion_matrix"]),
        ("CREMA-D_test", metrics["speaker_holdout"]["OAF"]["confusion_matrix"]),
        ("TESS_test", metrics["speaker_holdout"]["YAF"]["confusion_matrix"]),
    ):
        write_csv(OUT / f"source_data_confusion_{name}.csv", ["true/predicted", *LABELS],
                  [[label, *row] for label, row in zip(LABELS, matrix)])


def main() -> None:
    metrics = json.loads((SPEECH_DIR / "metrics.json").read_text(encoding="utf-8"))
    if "speaker_holdout" not in metrics:
        dataset_metrics = metrics["test_by_dataset"]
        metrics["speaker_holdout"] = {
            "OAF": dataset_metrics["CREMA-D"],
            "YAF": dataset_metrics["TESS"],
            "average": metrics["test"],
        }
    profile = load_model_profile()
    OUT.mkdir(parents=True, exist_ok=True)
    export_source_data(metrics, profile)
    figure_speech_performance(metrics, profile)
    figure_confusion_matrices(metrics)
    figure_model_profiles(profile)
    (OUT / "figure_metadata.json").write_text(
        json.dumps({"backend": "Python/matplotlib", "speech_metrics_source": "models/speech/metrics.json",
                    "profile": profile, "labels": LABELS}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"Research figures exported to: {OUT}")


if __name__ == "__main__":
    main()
