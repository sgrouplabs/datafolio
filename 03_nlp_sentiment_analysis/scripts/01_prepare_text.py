"""Prepare the IMDb dataset for LSTM training: load the tokenized top-10k
vocabulary, pad to a fixed 250 tokens, and split 80/10/10. The split is
persisted so training and evaluation see identical data."""

import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

NUM_WORDS = 10_000
MAXLEN = 250
SEED = 42


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    (x_train_raw, y_train_raw), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
        num_words=NUM_WORDS)

    x_all = np.concatenate([x_train_raw, x_test])
    y_all = np.concatenate([y_train_raw, y_test])

    # IMDb's own train/test split is already random, but reshuffle anyway so
    # the 80/10/10 split isn't sensitive to their ordering.
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(x_all))
    x_all, y_all = x_all[idx], y_all[idx]

    n = len(x_all)
    n_train, n_val = int(0.8 * n), int(0.1 * n)

    def pad(x):
        # post-padding + mask_zero in the model, so the LSTM skips pads.
        return pad_sequences(x, maxlen=MAXLEN, padding="post",
                             truncating="post")

    pad_train = pad(x_all[:n_train])
    pad_val = pad(x_all[n_train:n_train + n_val])
    pad_test = pad(x_all[n_train + n_val:])
    y_train = y_all[:n_train]
    y_val = y_all[n_train:n_train + n_val]
    y_test = y_all[n_train + n_val:]

    np.savez_compressed(os.path.join(RESULTS_DIR, "imdb_padded.npz"),
                        x_train=pad_train, y_train=y_train,
                        x_val=pad_val, y_val=y_val,
                        x_test=pad_test, y_test=y_test)

    lens = [len(r) for r in x_all[:n_train]]
    meta = {
        "num_words": NUM_WORDS,
        "maxlen": MAXLEN,
        "sizes": {"train": len(pad_train), "val": len(pad_val),
                  "test": len(pad_test)},
        "train_review_len_mean": float(np.mean(lens)),
        "train_review_len_p95": float(np.percentile(lens, 95)),
        "class_balance_train": float(np.mean(y_train)),
    }
    with open(os.path.join(RESULTS_DIR, "data_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
