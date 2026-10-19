"""NIH ChestX-ray14 loading: metadata filtering, patient-level splits, tf.data pipelines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit

from cxr.config import slug

METADATA_FILE = "Data_Entry_2017.csv"
TEST_LIST_FILE = "test_list.txt"


def load_metadata(data_dir: Path, classes: list[str]) -> pd.DataFrame:
    """Single-label images whose finding is one of `classes`, with resolved file paths.

    Multi-label images are dropped so the task is a clean multiclass problem.
    """
    df = pd.read_csv(data_dir / METADATA_FILE)
    df = df[~df["Finding Labels"].str.contains("|", regex=False)]
    df = df[df["Finding Labels"].isin(classes)].copy()

    # The release ships images across images_001/ ... images_012/ subfolders.
    images = {p.name: str(p) for p in data_dir.rglob("*.png")}
    df["path"] = df["Image Index"].map(images)
    df = df.dropna(subset=["path"])
    df["label"] = df["Finding Labels"].map({c: i for i, c in enumerate(classes)}).astype("int64")
    return df[["Image Index", "Patient ID", "path", "label"]].reset_index(drop=True)


def _group_split(df: pd.DataFrame, fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    keep, held_out = next(splitter.split(df, groups=df["Patient ID"]))
    return df.iloc[keep].reset_index(drop=True), df.iloc[held_out].reset_index(drop=True)


def split_by_patient(df: pd.DataFrame, data_dir: Path, val_fraction: float, seed: int):
    """Train/val/test with no patient appearing in more than one split.

    Uses the official test_list.txt when present (it is patient-disjoint), else a
    random 20% patient-level hold-out.
    """
    test_list = data_dir / TEST_LIST_FILE
    if test_list.exists():
        is_test = df["Image Index"].isin(set(test_list.read_text().split()))
        trainval, test = df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)
    else:
        trainval, test = _group_split(df, 0.2, seed)
    train, val = _group_split(trainval, val_fraction, seed)
    return train, val, test


def cap_per_class(df: pd.DataFrame, max_per_class: int | None, seed: int) -> pd.DataFrame:
    if not max_per_class:
        return df
    parts = [g.sample(min(len(g), max_per_class), random_state=seed) for _, g in df.groupby("label")]
    return pd.concat(parts).reset_index(drop=True)


def synthetic_frame(output_dir: Path, classes: list[str]) -> pd.DataFrame:
    """GAN-generated images written by `python -m cxr gan` (training split only)."""
    rows = []
    for label, name in enumerate(classes):
        for p in sorted((output_dir / "synthetic" / slug(name)).glob("*.png")):
            rows.append({"Image Index": p.name, "Patient ID": -1, "path": str(p), "label": label})
    return pd.DataFrame(rows, columns=["Image Index", "Patient ID", "path", "label"])


def class_weights(labels: np.ndarray, num_classes: int) -> dict[int, float]:
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return {i: float(w) for i, w in enumerate(weights)}


def load_image(path: tf.Tensor, image_size: int) -> tf.Tensor:
    """Grayscale float32 image in [0, 255], shape (size, size, 1)."""
    img = tf.io.decode_png(tf.io.read_file(path), channels=1)
    return tf.image.resize(tf.cast(img, tf.float32), [image_size, image_size])


def classifier_dataset(df: pd.DataFrame, num_classes: int, image_size: int, batch_size: int,
                       training: bool, seed: int = 0) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((df["path"].to_numpy(), df["label"].to_numpy()))
    if training:
        ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)

    def _load(path, label):
        img = load_image(path, image_size)
        if training:
            img = tf.image.random_brightness(img, 0.08 * 255)
            img = tf.image.random_contrast(img, 0.9, 1.1)
            img = tf.clip_by_value(img, 0.0, 255.0)
        img = tf.keras.applications.densenet.preprocess_input(tf.image.grayscale_to_rgb(img))
        return img, tf.one_hot(label, num_classes)

    return ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def gan_dataset(df: pd.DataFrame, image_size: int, batch_size: int, seed: int = 0) -> tf.data.Dataset:
    """(image in [-1, 1], label of shape (1,)) pairs for the conditional GAN."""
    ds = tf.data.Dataset.from_tensor_slices((df["path"].to_numpy(), df["label"].to_numpy().astype("int32")))
    ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)

    def _load(path, label):
        img = load_image(path, image_size) / 127.5 - 1.0
        return img, tf.reshape(label, (1,))

    return ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size, drop_remainder=True).prefetch(
        tf.data.AUTOTUNE)
