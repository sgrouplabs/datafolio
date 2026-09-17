"""Train the BiLSTM sentiment classifier.

Embedding(128, mask_zero) -> Bidirectional(LSTM(64)) -> Dense head with
sigmoid output. Checkpoints to models/lstm_sentiment.keras by val_loss.
"""

import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")

SEED = 42
EMBED_DIM = 128
LSTM_UNITS = 64
BATCH_SIZE = 128
EPOCHS = 15
INITIAL_LR = 1e-3


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    data = np.load(os.path.join(RESULTS_DIR, "imdb_padded.npz"))
    x_train, y_train = data["x_train"], data["y_train"]
    x_val, y_val = data["x_val"], data["y_val"]
    num_words, maxlen = int(data["x_train"].max()) + 1, x_train.shape[1]

    tf.random.set_seed(SEED)

    # input_dim=20000 rather than 10000: the embedding table is sized a bit
    # above the vocabulary cap, which costs nothing and avoids an off-by-one.
    model = models.Sequential([
        layers.Input(shape=(maxlen,)),
        layers.Embedding(input_dim=20000, output_dim=EMBED_DIM, mask_zero=True),
        # Bidirectional so polarity flips mid-review ("great cast, but...")
        # are scored with context from both directions.
        layers.Bidirectional(layers.LSTM(LSTM_UNITS)),
        layers.Dropout(0.5),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(INITIAL_LR),
                  loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                         restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(MODELS_DIR, "lstm_sentiment.keras"),
            monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=2, min_lr=1e-6),
    ]

    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                        epochs=EPOCHS, batch_size=BATCH_SIZE,
                        callbacks=callbacks)

    with open(os.path.join(RESULTS_DIR, "training_history.json"), "w") as f:
        json.dump({k: [float(v) for v in vals]
                   for k, vals in history.history.items()}, f, indent=2)

    test_loss, test_acc = model.evaluate(data["x_test"], data["y_test"])
    print(f"Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")
    with open(os.path.join(RESULTS_DIR, "test_metrics.json"), "w") as f:
        json.dump({"test_loss": float(test_loss),
                   "test_accuracy": float(test_acc)}, f, indent=2)


if __name__ == "__main__":
    main()
