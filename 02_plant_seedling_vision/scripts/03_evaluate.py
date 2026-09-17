"""Evaluate the trained seedling classifier on the hold-out split.

Rebuilds the same val/test split as training (same seed), loads the best
checkpoint, and writes the loss/accuracy curves plus confusion matrix to
results/.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


def main() -> None:
    with open(os.path.join(RESULTS_DIR, "training_history.json")) as f:
        hist = json.load(f)

    epochs = range(1, len(hist["loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))
    ax_loss.plot(epochs, hist["loss"], label="Training Loss")
    ax_loss.plot(epochs, hist["val_loss"], label="Validation Loss")
    ax_loss.set_title("Loss over Epochs")
    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Categorical Cross-Entropy")
    ax_loss.legend(); ax_loss.grid(alpha=0.3)

    ax_acc.plot(epochs, hist["accuracy"], label="Training Accuracy")
    ax_acc.plot(epochs, hist["val_accuracy"], label="Validation Accuracy")
    ax_acc.set_title("Accuracy over Epochs")
    ax_acc.set_xlabel("Epoch"); ax_acc.set_ylabel("Accuracy")
    ax_acc.legend(); ax_acc.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "training_history.png"), dpi=150)
    print("Saved results/training_history.png")

    # Must match training: same seed, same split, shuffle off so labels
    # line up with predictions for the confusion matrix.
    temp_ds = tf.keras.utils.image_dataset_from_directory(
        RAW_DIR, validation_split=0.30, subset="validation", seed=SEED,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="categorical",
        shuffle=False)
    n_batches = tf.data.experimental.cardinality(temp_ds).numpy()
    test_ds = temp_ds.skip(n_batches // 2)

    model = tf.keras.models.load_model(
        os.path.join(MODELS_DIR, "seedling_classifier.keras"))

    y_true = np.concatenate([np.argmax(y, axis=1) for _, y in test_ds])
    y_prob = model.predict(test_ds)
    y_pred = np.argmax(y_prob, axis=1)

    with open(os.path.join(RESULTS_DIR, "test_metrics.json")) as f:
        class_names = json.load(f)["class_names"]

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted Species"); ax.set_ylabel("Actual Species")
    ax.set_title("Seedling Classifier — Confusion Matrix (Hold-Out Test Set)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
    print("Saved results/confusion_matrix.png")

    print("\nClassification report:")
    labels = list(range(len(class_names)))
    print(classification_report(y_true, y_pred, labels=labels,
                                target_names=class_names, zero_division=0))
    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
        f.write(classification_report(y_true, y_pred, labels=labels,
                                      target_names=class_names,
                                      zero_division=0))


if __name__ == "__main__":
    main()
