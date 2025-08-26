# app/utils.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
from pathlib import Path

from config import ARTIFACTS_DIR


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def set_seeds(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass


def plot_confusion_matrix(cm, class_names, out_path, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True label', xlabel='Predicted label',
           title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def log_common_artifacts(run, y_true, y_pred, artifacts_subdir, class_names):
    path = ARTIFACTS_DIR / artifacts_subdir
    ensure_dir(path)
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, path / "confusion_matrix.png")
    summary = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "report": classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    }
    with open(path / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return path