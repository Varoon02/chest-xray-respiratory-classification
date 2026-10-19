"""DenseNet-121 transfer-learning classifier."""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

EMBEDDING_LAYER = "embedding"


def build_classifier(num_classes: int, image_size: int, pretrained: bool = True, dropout: float = 0.3):
    """Returns (model, backbone). The backbone starts frozen for head training."""
    backbone = keras.applications.DenseNet121(
        include_top=False,
        weights="imagenet" if pretrained else None,
        input_shape=(image_size, image_size, 3),
    )
    backbone.trainable = False

    inputs = keras.Input((image_size, image_size, 3))
    # training=False keeps BatchNorm statistics frozen even while fine-tuning,
    # which is more stable on small medical datasets.
    x = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name=EMBEDDING_LAYER)(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="probs")(x)
    return keras.Model(inputs, outputs, name="densenet121_cxr"), backbone


def compile_classifier(model: keras.Model, num_classes: int, lr: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc", multi_label=True, num_labels=num_classes),
        ],
    )


def embedding_model(model: keras.Model) -> keras.Model:
    """Maps images to the pooled 1024-d DenseNet features (input to the QSVM)."""
    return keras.Model(model.inputs, model.get_layer(EMBEDDING_LAYER).output)


def load_classifier(path) -> keras.Model:
    return keras.models.load_model(path, compile=False)
