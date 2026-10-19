"""Class-conditional DCGAN used to synthesise extra training images for minority classes."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_generator(latent_dim: int, num_classes: int, image_size: int) -> keras.Model:
    start = image_size // 16  # four stride-2 upsamplings
    z = keras.Input((latent_dim,), name="z")
    label = keras.Input((1,), dtype="int32", name="label")
    emb = layers.Flatten()(layers.Embedding(num_classes, 50)(label))

    x = layers.Concatenate()([z, emb])
    x = layers.Dense(start * start * 256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Reshape((start, start, 256))(x)
    for filters in (256, 128, 64, 32):
        x = layers.Conv2DTranspose(filters, 4, strides=2, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
    image = layers.Conv2D(1, 3, padding="same", activation="tanh")(x)
    return keras.Model([z, label], image, name="generator")


def build_discriminator(num_classes: int, image_size: int) -> keras.Model:
    image = keras.Input((image_size, image_size, 1), name="image")
    label = keras.Input((1,), dtype="int32", name="label")
    # The label becomes an extra image channel so every conv layer sees it.
    label_map = layers.Reshape((image_size, image_size, 1))(layers.Embedding(num_classes, image_size * image_size)(label))

    x = layers.Concatenate()([image, label_map])
    for filters in (32, 64, 128, 256):
        x = layers.Conv2D(filters, 4, strides=2, padding="same")(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.3)(x)
    logit = layers.Dense(1)(layers.Flatten()(x))
    return keras.Model([image, label], logit, name="discriminator")


class ConditionalGAN:
    def __init__(self, num_classes: int, image_size: int, latent_dim: int = 128, lr: float = 2e-4):
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.generator = build_generator(latent_dim, num_classes, image_size)
        self.discriminator = build_discriminator(num_classes, image_size)
        self.g_opt = keras.optimizers.Adam(lr, beta_1=0.5)
        self.d_opt = keras.optimizers.Adam(lr, beta_1=0.5)
        self.bce = keras.losses.BinaryCrossentropy(from_logits=True)

    @tf.function
    def train_step(self, images, labels):
        z = tf.random.normal((tf.shape(images)[0], self.latent_dim))
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            fake = self.generator([z, labels], training=True)
            real_logits = self.discriminator([images, labels], training=True)
            fake_logits = self.discriminator([fake, labels], training=True)
            # One-sided label smoothing (real = 0.9) keeps the discriminator from saturating.
            d_loss = (self.bce(tf.fill(tf.shape(real_logits), 0.9), real_logits)
                      + self.bce(tf.zeros_like(fake_logits), fake_logits))
            g_loss = self.bce(tf.ones_like(fake_logits), fake_logits)

        d_vars = self.discriminator.trainable_variables
        g_vars = self.generator.trainable_variables
        self.d_opt.apply_gradients(zip(d_tape.gradient(d_loss, d_vars), d_vars))
        self.g_opt.apply_gradients(zip(g_tape.gradient(g_loss, g_vars), g_vars))
        return d_loss, g_loss

    def generate(self, class_index: int, n: int, batch_size: int = 64, seed: int | None = None) -> np.ndarray:
        """Returns n uint8 grayscale images of shape (n, H, W)."""
        rng = np.random.default_rng(seed)
        out = []
        for start in range(0, n, batch_size):
            b = min(batch_size, n - start)
            z = rng.standard_normal((b, self.latent_dim)).astype("float32")
            labels = np.full((b, 1), class_index, dtype="int32")
            imgs = self.generator([z, labels], training=False).numpy()
            out.append(np.clip((imgs[..., 0] + 1.0) * 127.5, 0, 255).astype("uint8"))
        return np.concatenate(out) if out else np.zeros((0, 0, 0), dtype="uint8")
