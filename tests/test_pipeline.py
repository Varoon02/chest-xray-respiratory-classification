"""End-to-end smoke test on a tiny synthetic dataset laid out like NIH ChestX-ray14.

Exercises every stage (GAN -> synthetic images -> classifier -> evaluation -> QSVM)
in seconds on CPU, so CI catches breakage without the 45 GB dataset.
"""

import json

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
import yaml

from cxr import pipeline
from cxr.config import load_config
from cxr.data import class_weights, load_metadata, split_by_patient
from cxr.metrics import evaluate_scores

CLASSES = ["No Finding", "Effusion", "Pneumonia"]
PER_CLASS = 24


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    root = tmp_path_factory.mktemp("cxr")
    data = root / "data"
    images = data / "images_001" / "images"
    images.mkdir(parents=True)
    rng = np.random.default_rng(0)

    rows, test_names = [], []
    for label, name in enumerate(CLASSES):
        for i in range(PER_CLASS):
            fname = f"{label:02d}{i:04d}_000.png"
            # Class-dependent brightness so there is something learnable.
            img = np.clip(rng.normal(60 + 60 * label, 20, (32, 32)), 0, 255).astype("uint8")
            tf.io.write_file(str(images / fname), tf.io.encode_png(img[..., None]))
            rows.append({"Image Index": fname, "Finding Labels": name, "Patient ID": label * 1000 + i})
            if i < 6:
                test_names.append(fname)
    # Multi-label and out-of-scope rows must be filtered out.
    rows.append({"Image Index": "x.png", "Finding Labels": "Effusion|Pneumonia", "Patient ID": 9})
    rows.append({"Image Index": "y.png", "Finding Labels": "Hernia", "Patient ID": 9})
    pd.DataFrame(rows).to_csv(data / "Data_Entry_2017.csv", index=False)
    (data / "test_list.txt").write_text("\n".join(test_names))

    config = {
        "data_dir": str(data), "output_dir": str(root / "outputs"), "classes": CLASSES,
        "image_size": 32, "val_fraction": 0.2, "seed": 0,
        "classifier": {"pretrained": False, "batch_size": 8, "max_per_class": None, "head_epochs": 1,
                       "head_lr": 1e-3, "finetune_epochs": 1, "finetune_lr": 1e-4},
        "gan": {"image_size": 32, "latent_dim": 16, "batch_size": 8, "epochs": 1, "lr": 2e-4,
                "synthetic_per_class": {"Pneumonia": 10}},
        "qsvm": {"n_qubits": 2, "reps": 1, "C": 1.0, "train_samples": 30, "test_samples": 18},
    }
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return load_config(path)


def test_metadata_and_patient_split(cfg):
    df = load_metadata(cfg["data_dir"], CLASSES)
    assert len(df) == len(CLASSES) * PER_CLASS
    train, val, test = split_by_patient(df, cfg["data_dir"], 0.2, 0)
    assert len(test) == 6 * len(CLASSES)
    ids = [set(s["Patient ID"]) for s in (train, val, test)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])


def test_class_weights_balance():
    w = class_weights(np.array([0, 0, 0, 1]), 2)
    assert w[1] == pytest.approx(3 * w[0])


def test_metrics_handle_missing_class():
    y = np.array([0, 0, 1, 1])
    probs = np.array([[0.9, 0.1, 0], [0.8, 0.2, 0], [0.3, 0.7, 0], [0.1, 0.9, 0]])
    r = evaluate_scores(y, probs, ["a", "b", "c"])
    assert r["macro_auc"] == 1.0 and set(r["per_class_auc"]) == {"a", "b"}
    assert r["accuracy"] == 1.0


def test_full_pipeline(cfg):
    pipeline.train_gan(cfg)
    synthetic = list((cfg["output_dir"] / "synthetic" / "Pneumonia").glob("*.png"))
    assert len(synthetic) == 10

    for use_synthetic in (False, True):
        pipeline.train_classifier(cfg, use_synthetic)
        result = pipeline.evaluate(cfg, use_synthetic)
        assert 0.0 <= result["macro_auc"] <= 1.0
        assert result["n"] == 6 * len(CLASSES)

    q = pipeline.qsvm(cfg, use_synthetic=True)
    assert q["train_samples"] == 30
    for key in ("quantum_svm", "classical_rbf_svm"):
        assert 0.0 <= q[key]["macro_auc"] <= 1.0
    saved = json.loads((cfg["output_dir"] / "qsvm_gan.json").read_text())
    assert saved["n_qubits"] == 2
