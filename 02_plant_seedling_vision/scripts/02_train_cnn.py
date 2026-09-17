"""Train a MobileNetV2 transfer-learning classifier on the seedling dataset.

Frozen ImageNet backbone + small head, with in-graph augmentation.
Splits 70/15/15 and checkpoints the best model by val_loss.
"""

import json
import os

import tensorflow as tf
from tensorflow.keras import layers

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
EPOCHS = 30
INITIAL_LR = 1e-3


def build_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        RAW_DIR, validation_split=0.30, subset="training", seed=SEED,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="categorical")
    temp_ds = tf.keras.utils.image_dataset_from_directory(
        RAW_DIR, validation_split=0.30, subset="validation", seed=SEED,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="categorical")

    class_names = train_ds.class_names

    # Split the 30% pool into 15% val / 15% test.
    n_batches = tf.data.experimental.cardinality(temp_ds).numpy()
    val_ds = temp_ds.take(n_batches // 2)
    test_ds = temp_ds.skip(n_batches // 2)

    AUTOTUNE = tf.data.AUTOTUNE
    return (train_ds.prefetch(AUTOTUNE), val_ds.prefetch(AUTOTUNE),
            test_ds.prefetch(AUTOTUNE), class_names)


def build_model(num_classes: int) -> tf.keras.Model:
    # No vertical flips: seedlings grow upward, so flipped-vertical images
    # aren't something the model would ever see in the field.
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.1, 0.1),
    ], name="data_augmentation")

    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
    base.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)  # keep BatchNorm in inference mode
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    train_ds, val_ds, test_ds, class_names = build_datasets()
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    model = build_model(num_classes)
    model.compile(optimizer=tf.keras.optimizers.Adam(INITIAL_LR),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                         restore_best_weights=True),
        # .keras format — the legacy .h5 path fails to load on recent Keras
        # (MobileNetV2's TrueDivide layer doesn't round-trip).
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(MODELS_DIR, "seedling_classifier.keras"),
            monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=3, min_lr=1e-6),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
                        callbacks=callbacks)

    with open(os.path.join(RESULTS_DIR, "training_history.json"), "w") as f:
        json.dump({k: [float(v) for v in vals]
                   for k, vals in history.history.items()}, f, indent=2)

    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")
    with open(os.path.join(RESULTS_DIR, "test_metrics.json"), "w") as f:
        json.dump({"test_loss": float(test_loss),
                   "test_accuracy": float(test_acc),
                   "class_names": class_names}, f, indent=2)


if __name__ == "__main__":
    main()
