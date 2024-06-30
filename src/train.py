import logging
from functools import partial
from pathlib import Path
from time import strftime
from typing import Any, Callable, Dict, List, Tuple, cast

import hydra
import keras
import keras_cv
import keras_tuner as kt
import numpy as np
import tensorflow as tf
from omegaconf import DictConfig, ListConfig

from datasets.deep_weeds import get_train_val_dataloader

log = logging.getLogger(__name__)


def get_model(
    seed: int,
    input_shape: Tuple[int],
    n_classes: int,
    blocks: List[Dict],
    output_activation: str,
) -> keras.Model:
    MODEL_NAME = "MeNet"

    tf.random.set_seed(seed)

    def conv_skip_block(x, filters: int, kernel_shape=(3, 3), stride=(1, 1), padding="same"):
        skip = x
        x = tf.keras.layers.Conv2D(filters, kernel_shape, padding=padding)(x)
        x = tf.keras.layers.Conv2D(filters, kernel_shape, padding=padding)(x)
        x = tf.keras.layers.Concatenate()([x, skip])
        x = keras.layers.BatchNormalization()(x)
        return x

    # input_shape = (256,256,3)
    input_ = tf.keras.Input(shape=input_shape)
    input_layer = tf.keras.layers.Conv2D(64, (7, 7), (2, 2), padding="same")
    pool_layer = keras.layers.MaxPool2D(pool_size=(2, 2))
    global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
    output_layer = tf.keras.layers.Dense(n_classes, activation="softmax")

    # Input conv + pool
    x = input_layer(input_)
    x = pool_layer(x)

    # Residual blocks
    for block in blocks:
        x = conv_skip_block(x, **block)

    # Output pool + fully connected
    x = global_avg_pool(x)
    output = output_layer(x)

    return tf.keras.Model(name=MODEL_NAME, inputs=[input_], outputs=[output])


preprocessor = keras_cv.layers.Augmenter(
    [
        keras_cv.layers.Rescaling(scale=1.0 / 255),
    ]
)

augmenter = keras_cv.layers.Augmenter(
    [
        # keras_cv.layers.Rescaling(scale=1.0 / 255),
    ]
)


def preprocess_data(images, labels, augment=False):
    labels = tf.one_hot(labels, 10)
    inputs = {"images": images, "labels": labels}
    outputs = inputs
    # TODO: add simple augmentations beyond scaling
    # TODO: make another notebook visualising effect of some transforms
    outputs = preprocessor(outputs)

    if augment:
        outputs = augmenter(outputs)

    return outputs["images"], outputs["labels"]


def train(
    loss: str,
    optimizer: str,
    optimizer_params: DictConfig,
    metrics: List[str],
    batch_size: int,
    train_args: DictConfig,
    tensorboard_path: Path,
    model_cfg: DictConfig,
    class_weight: DictConfig,
    f_get_model: Callable[..., keras.Model],
    f_get_dataloader: Callable[..., tf.data.Dataset],
) -> Tuple[keras.Model, Dict]:

    train_data, val_data = f_get_dataloader()
    assert isinstance(train_data, tf.data.Dataset)
    assert isinstance(val_data, tf.data.Dataset)
    train_data = (
        train_data.batch(batch_size)
        .map(lambda x, y: preprocess_data(x, y, augment=True), num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_data = (
        val_data.batch(batch_size)
        .map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    model = f_get_model(**model_cfg)  # type: ignore
    assert isinstance(model, keras.Model)
    model.compile(
        loss=loss,
        optimizer=getattr(tf.keras.optimizers, optimizer)(**optimizer_params),
        metrics=list(metrics) if isinstance(metrics, ListConfig) else metrics,
    )

    history = model.fit(
        x=train_data,
        validation_data=val_data,
        callbacks=[
            keras.callbacks.TensorBoard(log_dir=tensorboard_path),
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        ],
        class_weight={k: v / class_weight.weight_divisor for (k, v) in class_weight.weight_map.items()},
        **train_args,
    )

    return model, history


@hydra.main(config_path="../conf", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    seed = cfg.seed
    run_logdir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    model_cfg = cfg.model
    train_path = Path(cfg.data.dataset)

    get_dataloader = partial(
        get_train_val_dataloader, train_path=train_path, seed=seed, val_split=cfg.data.val_split
    )

    if cfg.print_summary:
        model = get_model(**model_cfg)
        model.summary(print_fn=log.info)
        del model

    model, history = train(
        tensorboard_path=run_logdir,
        model_cfg=model_cfg,
        f_get_model=get_model,
        f_get_dataloader=get_dataloader,
        **cfg.train,
    )

    # TODO: add final validation plots here


if __name__ == "__main__":
    main()
