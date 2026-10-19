"""The four experiment stages: GAN training, classifier training, evaluation, QSVM."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from cxr.config import slug
from cxr.data import (cap_per_class, class_weights, classifier_dataset, gan_dataset, load_metadata,
                      split_by_patient, synthetic_frame)
from cxr.gan import ConditionalGAN
from cxr.metrics import evaluate_scores
from cxr.models import build_classifier, compile_classifier, embedding_model, load_classifier


def _splits(cfg: dict):
    df = load_metadata(cfg["data_dir"], cfg["classes"])
    return split_by_patient(df, cfg["data_dir"], cfg["val_fraction"], cfg["seed"])


def _tag(use_synthetic: bool) -> str:
    return "gan" if use_synthetic else "baseline"


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {path}")


# --------------------------------------------------------------------------- GAN

def train_gan(cfg: dict) -> None:
    g = cfg["gan"]
    classes = cfg["classes"]
    train, _, _ = _splits(cfg)  # GAN only ever sees training patients
    train = cap_per_class(train, cfg["classifier"]["max_per_class"], cfg["seed"])
    ds = gan_dataset(train, g["image_size"], g["batch_size"], cfg["seed"])

    gan = ConditionalGAN(len(classes), g["image_size"], g["latent_dim"], g["lr"])
    for epoch in range(1, g["epochs"] + 1):
        start, d_losses, g_losses = time.time(), [], []
        for images, labels in ds:
            d_loss, g_loss = gan.train_step(images, labels)
            d_losses.append(float(d_loss))
            g_losses.append(float(g_loss))
        print(f"gan epoch {epoch}/{g['epochs']}: d_loss {np.mean(d_losses):.3f} "
              f"g_loss {np.mean(g_losses):.3f} ({time.time() - start:.0f}s)")

    out = cfg["output_dir"]
    (out / "gan").mkdir(exist_ok=True)
    gan.generator.save_weights(str(out / "gan" / "generator.weights.h5"))

    for name, count in (g.get("synthetic_per_class") or {}).items():
        target = out / "synthetic" / slug(name)
        target.mkdir(parents=True, exist_ok=True)
        images = gan.generate(classes.index(name), int(count), seed=cfg["seed"])
        for i, img in enumerate(images):
            tf.io.write_file(str(target / f"syn_{i:05d}.png"), tf.io.encode_png(img[..., None]))
        print(f"generated {len(images)} synthetic '{name}' images -> {target}")


# -------------------------------------------------------------------- classifier

def train_classifier(cfg: dict, use_synthetic: bool) -> None:
    c = cfg["classifier"]
    classes, size, n = cfg["classes"], cfg["image_size"], len(cfg["classes"])
    train, val, _ = _splits(cfg)
    train = cap_per_class(train, c["max_per_class"], cfg["seed"])
    if use_synthetic:
        synthetic = synthetic_frame(cfg["output_dir"], classes)
        if synthetic.empty:
            raise SystemExit("no synthetic images found; run `python -m cxr gan` first")
        train = pd.concat([train, synthetic], ignore_index=True)
    print(f"train {len(train)} (per class {np.bincount(train['label'], minlength=n).tolist()}), "
          f"val {len(val)}")

    train_ds = classifier_dataset(train, n, size, c["batch_size"], training=True, seed=cfg["seed"])
    val_ds = classifier_dataset(val, n, size, c["batch_size"], training=False)
    weights = class_weights(train["label"].to_numpy(), n)

    model, backbone = build_classifier(n, size, pretrained=c["pretrained"])
    path = cfg["output_dir"] / f"classifier_{_tag(use_synthetic)}.keras"
    callbacks = [keras.callbacks.ModelCheckpoint(str(path), monitor="val_auc", mode="max", save_best_only=True)]

    # Phase 1: train the new head on frozen ImageNet features.
    compile_classifier(model, n, c["head_lr"])
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=c["head_epochs"], class_weight=weights,
                   callbacks=callbacks, verbose=2)
    # Phase 2: unfreeze and fine-tune the whole network at a lower learning rate.
    backbone.trainable = True
    compile_classifier(model, n, c["finetune_lr"])
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=c["finetune_epochs"], class_weight=weights,
                   callbacks=callbacks + [keras.callbacks.EarlyStopping("val_auc", mode="max", patience=4,
                                                                         restore_best_weights=True)],
                   verbose=2)
    if not path.exists():  # val AUC never improved past NaN (tiny val sets); keep the final model
        model.save(str(path))

    history = {k: h1.history.get(k, []) + h2.history.get(k, []) for k in set(h1.history) | set(h2.history)}
    _write_json(cfg["output_dir"] / f"history_{_tag(use_synthetic)}.json", history)


# -------------------------------------------------------------------- evaluation

def evaluate(cfg: dict, use_synthetic: bool) -> dict:
    classes, n = cfg["classes"], len(cfg["classes"])
    _, _, test = _splits(cfg)
    model = load_classifier(cfg["output_dir"] / f"classifier_{_tag(use_synthetic)}.keras")
    ds = classifier_dataset(test, n, cfg["image_size"], cfg["classifier"]["batch_size"], training=False)
    probs = model.predict(ds, verbose=0)
    result = evaluate_scores(test["label"].to_numpy(), probs, classes)
    _write_json(cfg["output_dir"] / f"metrics_{_tag(use_synthetic)}.json", result)
    print(f"[{_tag(use_synthetic)}] test macro AUC {result['macro_auc']:.4f}, accuracy {result['accuracy']:.4f}")
    for name, auc in result["per_class_auc"].items():
        print(f"  {name:<14} AUC {auc:.4f}")
    return result


# -------------------------------------------------------------------------- QSVM

def _stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n >= len(df):
        return df
    sample, _ = train_test_split(df, train_size=n, stratify=df["label"], random_state=seed)
    return sample.reset_index(drop=True)


def _embed(model, df: pd.DataFrame, cfg: dict) -> np.ndarray:
    ds = classifier_dataset(df, len(cfg["classes"]), cfg["image_size"], cfg["classifier"]["batch_size"],
                            training=False)
    return embedding_model(model).predict(ds, verbose=0)


def qsvm(cfg: dict, use_synthetic: bool) -> dict:
    """Quantum-kernel SVM vs. classical RBF SVM on PCA-reduced DenseNet embeddings."""
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_machine_learning.kernels import FidelityQuantumKernel

    q = cfg["qsvm"]
    classes, seed = cfg["classes"], cfg["seed"]
    train, _, test = _splits(cfg)
    # Kernel cost grows with n^2 circuit evaluations, so work on a stratified subsample.
    train = _stratified_sample(train, q["train_samples"], seed)
    test = _stratified_sample(test, q["test_samples"], seed)

    model = load_classifier(cfg["output_dir"] / f"classifier_{_tag(use_synthetic)}.keras")
    x_train, x_test = _embed(model, train, cfg), _embed(model, test, cfg)
    y_train, y_test = train["label"].to_numpy(), test["label"].to_numpy()

    # 1024-d features -> n_qubits dims, scaled to rotation angles in [0, pi].
    scaler, pca, angles = StandardScaler(), PCA(n_components=q["n_qubits"], random_state=seed), MinMaxScaler((0, np.pi))
    x_train = angles.fit_transform(pca.fit_transform(scaler.fit_transform(x_train)))
    x_test = np.clip(angles.transform(pca.transform(scaler.transform(x_test))), 0, np.pi)

    feature_map = ZZFeatureMap(feature_dimension=q["n_qubits"], reps=q["reps"], entanglement="linear")
    kernel = FidelityQuantumKernel(feature_map=feature_map)
    start = time.time()
    k_train = kernel.evaluate(x_vec=x_train)
    k_test = kernel.evaluate(x_vec=x_test, y_vec=x_train)
    kernel_seconds = time.time() - start

    quantum = SVC(kernel="precomputed", C=q["C"], probability=True, random_state=seed).fit(k_train, y_train)
    classical = SVC(kernel="rbf", C=q["C"], probability=True, random_state=seed).fit(x_train, y_train)

    result = {
        "n_qubits": q["n_qubits"],
        "reps": q["reps"],
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "kernel_seconds": round(kernel_seconds, 1),
        "quantum_svm": evaluate_scores(y_test, _full_proba(quantum, k_test, len(classes)), classes),
        "classical_rbf_svm": evaluate_scores(y_test, _full_proba(classical, x_test, len(classes)), classes),
    }
    _write_json(cfg["output_dir"] / f"qsvm_{_tag(use_synthetic)}.json", result)
    for name in ("quantum_svm", "classical_rbf_svm"):
        r = result[name]
        print(f"{name:<18} macro AUC {r['macro_auc']:.4f}  accuracy {r['accuracy']:.4f}")
    return result


def _full_proba(svm: SVC, x, num_classes: int) -> np.ndarray:
    """predict_proba padded to all classes (a subsample can miss a class)."""
    probs = np.zeros((len(x), num_classes))
    probs[:, svm.classes_] = svm.predict_proba(x)
    return probs
