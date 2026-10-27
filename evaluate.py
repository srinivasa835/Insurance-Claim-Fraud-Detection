"""Shared metrics and plots (accuracy/loss curves, confusion matrix, predictions)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             roc_curve)


def summarize(y_true, y_pred, y_prob=None):
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None and len(set(y_true)) > 1:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
    return out


def save_report(name, y_true, y_pred, y_prob, texts, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    metrics = summarize(y_true, y_pred, y_prob)
    with open(os.path.join(out_dir, f"{name}_report.txt"), "w") as f:
        f.write(classification_report(y_true, y_pred, target_names=["genuine", "fraud"], digits=4))
        f.write("\n" + "\n".join(f"{k}: {v:.4f}" for k, v in metrics.items()) + "\n")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["genuine", "fraud"], yticklabels=["genuine", "fraud"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"{name}: confusion matrix")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{name}_confusion_matrix.png"), dpi=200)
    plt.close()

    if y_prob is not None and len(set(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure(figsize=(4.5, 4))
        plt.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.3f}")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
        plt.title(f"{name}: ROC curve"); plt.legend()
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{name}_roc.png"), dpi=200)
        plt.close()

    preds = pd.DataFrame({"narrative": texts, "actual": y_true, "predicted": y_pred,
                          "fraud_probability": y_prob})
    preds.head(50).to_csv(os.path.join(out_dir, f"{name}_sample_predictions.csv"), index=False)
    return metrics


def plot_curves(history, name, out_dir):
    """history: dict with lists train_loss, val_loss, train_acc, val_acc."""
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].plot(epochs, history["train_loss"], marker="o", label="train")
    ax[0].plot(epochs, history["val_loss"], marker="o", label="validation")
    ax[0].set_title("Loss"); ax[0].set_xlabel("Epoch"); ax[0].legend()
    ax[1].plot(epochs, history["train_acc"], marker="o", label="train")
    ax[1].plot(epochs, history["val_acc"], marker="o", label="validation")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("Epoch"); ax[1].legend()
    plt.suptitle(name); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_curves.png"), dpi=200)
    plt.close()
