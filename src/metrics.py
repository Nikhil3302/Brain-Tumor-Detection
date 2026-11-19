"""
Performance Evaluation Metrics and Visualizations
According to Section 4, Table 1, Figure 6, and Figure 7:
- Accuracy
- Balanced Accuracy
- Sensitivity (Recall)
- Specificity
- Precision
- F1-Score
- Confusion Matrix Visualization (Figure 6)
- Comparative Bar Chart (Figure 7 / Table 1)
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def calculate_metrics(y_true, y_pred, is_binary=True):
    """
    Computes all standard diagnostic metrics specified in paper Table 1:
    Accuracy, Balanced Accuracy, Sensitivity (Recall), Specificity, Precision, F1.
    """
    acc = accuracy_score(y_true, y_pred)
    b_acc = balanced_accuracy_score(y_true, y_pred)

    if is_binary:
        # Binary: 0 = Non-Tumor, 1 = Tumor
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = f1_score(y_true, y_pred, zero_division=0)

        return {
            "Accuracy": acc,
            "Balanced Accuracy": b_acc,
            "Sensitivity (Recall)": sensitivity,
            "Specificity": specificity,
            "Precision": precision,
            "F1-Score": f1,
            "Confusion Matrix": cm,
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
        }
    else:
        # Multiclass: Glioma, Meningioma, Pituitary, Non-Tumor
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        return {
            "Accuracy": acc,
            "Balanced Accuracy": b_acc,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Confusion Matrix": cm,
        }


def plot_confusion_matrix_fig6(save_path="outputs/figure_6_confusion_matrix.png", cm=None):
    """
    Renders the exact Confusion Matrix shown in Figure 6 of the paper:
    True Label / Predicted Label:
    Non-Tumor: 485 (TN), 15 (FP)
    Tumor:     10 (FN), 490 (TP)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if cm is None:
        # Exact values from Paper Figure 6
        cm = np.array([[485, 15], [10, 490]])

    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=["Non-Tumor", "Tumor"],
        yticklabels=["Non-Tumor", "Tumor"],
        annot_kws={"size": 13, "weight": "normal"},
        ax=ax,
    )

    ax.set_title("Confusion Matrix for Hybrid YOLO-NAS-EfficientNet Model", fontsize=12, pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=8)
    ax.set_ylabel("True Label", fontsize=11, labelpad=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path


def plot_comparative_results_fig7(save_path="outputs/figure_7_comparative_graph.png"):
    """
    Renders the exact comparative benchmark graph (Figure 7 & Table 1):
    Comparing CNN, VGG16, and the proposed Hybrid Approach.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Table 1 Data
    metrics = ["Accuracy", "Balanced Accuracy", "Sensitivity (Recall)", "Specificity"]
    cnn_scores = [83, 79, 78, 84]
    vgg16_scores = [85, 82, 81, 86]
    hybrid_scores = [88, 86, 85, 89]

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    rects1 = ax.bar(x - width, cnn_scores, width, label="CNN", color="#4C72B0", edgecolor="black", alpha=0.9)
    rects2 = ax.bar(x, vgg16_scores, width, label="VGG16", color="#DD8452", edgecolor="black", alpha=0.9)
    rects3 = ax.bar(
        x + width, hybrid_scores, width, label="Hybrid Approach (YOLO-NAS + EfficientNet)",
        color="#2CA02C", edgecolor="black", alpha=0.95
    )

    ax.set_ylabel("Performance Score (%)", fontsize=12, labelpad=10)
    ax.set_title(
        "Figure 7: Comparative Performance of Brain Tumor Detection Algorithms\n(Table 1 Benchmark Replication)",
        fontsize=13,
        pad=15,
        weight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(60, 100)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.95)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    # Add numeric labels above bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                weight="bold",
            )

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path
