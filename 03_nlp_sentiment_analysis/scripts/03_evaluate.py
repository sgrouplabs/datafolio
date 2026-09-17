"""Evaluate the BiLSTM sentiment model on the hold-out test set and write
the loss/accuracy curves and confusion matrix to results/."""

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
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def main() -> None:
    with open(os.path.join(RESULTS_DIR, "training_history.json")) as f:
        hist = json.load(f)

    epochs = range(1, len(hist["loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))
    ax_loss.plot(epochs, hist["loss"], label="Training Loss")
    ax_loss.plot(epochs, hist["val_loss"], label="Validation Loss")
    ax_loss.set_title("Loss over Epochs")
    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Binary Cross-Entropy")
    ax_loss.legend(); ax_loss.grid(alpha=0.3)

    ax_acc.plot(epochs, hist["accuracy"], label="Training Accuracy")
    ax_acc.plot(epochs, hist["val_accuracy"], label="Validation Accuracy")
    ax_acc.set_title("Accuracy over Epochs")
    ax_acc.set_xlabel("Epoch"); ax_acc.set_ylabel("Accuracy")
    ax_acc.legend(); ax_acc.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "training_history.png"), dpi=150)
    print("Saved results/training_history.png")

    data = np.load(os.path.join(RESULTS_DIR, "imdb_padded.npz"))
    x_test, y_test = data["x_test"], data["y_test"]

    model = tf.keras.models.load_model(
        os.path.join(MODELS_DIR, "lstm_sentiment.keras"))

    proba = model.predict(x_test, batch_size=256).ravel()
    y_pred = (proba >= 0.5).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred: Negative", "Pred: Positive"],
                yticklabels=["Actual: Negative", "Actual: Positive"], ax=ax)
    ax.set_title("BiLSTM Sentiment — Confusion Matrix (Hold-Out Test Set)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
    print("Saved results/confusion_matrix.png")

    report = classification_report(y_test, y_pred,
                                   target_names=["Negative", "Positive"])
    print("\nClassification report:")
    print(report)
    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
        f.write(report)


if __name__ == "__main__":
    main()
