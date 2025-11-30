import numpy as np
from evaluate import load as load_metric
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import mlflow
import seaborn as sns
import os

acc_metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = acc_metric.compute(predictions=preds, references=labels)["accuracy"]
    precision = precision_score(labels, preds, average="weighted")
    recall = recall_score(labels, preds, average="weighted")
    f1 = f1_score(labels, preds, average="weighted")

    metrics_report = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    precision_cls, recall_cls, f1_cls, support_cls = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
    for i in range(len(precision_cls)):
        metrics_report[f"precision_class_{i}"] = precision_cls[i]
        metrics_report[f"recall_class_{i}"] = recall_cls[i]
        metrics_report[f"f1_class_{i}"] = f1_cls[i]
        metrics_report[f"support_class_{i}"] = support_cls[i]

    return metrics_report

def log_confusion_matrix(labels, preds, class_names):
    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=class_names,
                yticklabels=class_names,
                cmap="Blues",
                ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()

    mlflow.log_figure(fig, "confusion_matrix.png")

    plt.close(fig)
