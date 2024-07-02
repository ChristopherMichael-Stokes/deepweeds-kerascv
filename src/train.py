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
    input_shape: Tuple[int],
    n_classes: int,
    blocks: List[Dict],
    output_activation: str,
    seed: int,
    block_dropout: float | None = None,
    scale_inputs: bool | None = None,
) -> keras.Model:
    MODEL_NAME = "MeNet"

    tf.random.set_seed(seed)

    def conv_skip_block(
        x: tf.Tensor, filters: int, kernel_shape=(3, 3), padding="same", drop_rate=None
    ) -> tf.Tensor:
        """Implements a residual block basically the same as resnet - https://arxiv.org/pdf/1512.03385,
        however with spatial dropout for improved training stability."""
        skip = x
        x = tf.keras.layers.Conv2D(filters, kernel_shape, padding=padding)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        if drop_rate:
            x = keras.layers.SpatialDropout2D(drop_rate)(x)
        x = tf.keras.layers.Conv2D(filters, kernel_shape, padding=padding)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Concatenate()([x, skip])
        x = tf.keras.layers.ReLU()(x)
        return x

    input_ = tf.keras.Input(shape=input_shape)
    scaling = keras_cv.layers.Rescaling(scale=1.0 / 255)
    input_layer = tf.keras.layers.Conv2D(64, (7, 7), (2, 2), padding="same")
    pool_layer = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
    global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
    output_layer = tf.keras.layers.Dense(n_classes, activation=output_activation)

    # Input conv + pool
    x = input_layer(input_)
    if scale_inputs:
        x = scaling(x)
    x = pool_layer(x)

    # Residual blocks
    for block in blocks:
        x = conv_skip_block(x, drop_rate=block_dropout, **block)

    # Output pool + fully connected
    x = global_avg_pool(x)
    x = tf.keras.layers.BatchNormalization()(x)
    output = output_layer(x)

    return tf.keras.Model(name=MODEL_NAME, inputs=[input_], outputs=[output])


def get_preprocessors():
    # TODO: add simple augmentations + visualise outputs in a notebook,
    # One hunch is that a saturation augment would help as there seems to be
    # a lot of variation between image samples or image locations

    jitter = keras_cv.layers.RandomColorJitter(
        value_range=(0, 255),
        brightness_factor=(-0.2, 0.2),
        contrast_factor=(0.5, 0.7),
        saturation_factor=(0.4, 0.6),
        hue_factor=(0.0, 0.3),
    )
    cut_mix = keras_cv.layers.Augmenter(
        [
            keras_cv.layers.CutMix(),
            keras_cv.layers.MixUp(),
        ]
    )
    return keras_cv.layers.RandomAugmentationPipeline(
        [
            jitter,
            # cut_mix,
        ],
        augmentations_per_image=1,
    )


def preprocess_data(
    images,
    labels,
    augmenter: keras_cv.layers.Augmenter | None = None,
):
    labels = tf.one_hot(labels, 10, dtype=tf.bfloat16)
    inputs = {"images": images, "labels": labels}

    outputs = inputs

    if augmenter:
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
    augment: bool,
    seed: int,
    f_get_model: Callable[..., keras.Model],
    f_get_dataloader: Callable[..., tf.data.Dataset],
    loss_params: DictConfig | None = None,
) -> Tuple[keras.Model, Dict]:
    tf.keras.utils.set_random_seed(seed)

    train_data, val_data = f_get_dataloader()
    assert isinstance(train_data, tf.data.Dataset)
    assert isinstance(val_data, tf.data.Dataset)

    augmenter = get_preprocessors() if augment else None
    log.info(f"Augmenter pipeline: {augmenter}")

    train_data = (
        train_data.batch(batch_size)
        .map(
            lambda x, y: preprocess_data(x, y, augmenter),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .prefetch(tf.data.AUTOTUNE)
    )

    val_data = (
        val_data.batch(batch_size)
        .map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    f_loss = getattr(tf.keras.losses, loss)
    if loss_params:
        f_loss = partial(f_loss, **loss_params)

    model = f_get_model(**model_cfg)  # type: ignore
    assert isinstance(model, keras.Model)
    model.compile(
        loss=f_loss,
        optimizer=getattr(tf.keras.optimizers, optimizer)(**optimizer_params),
        metrics=list(metrics) if isinstance(metrics, ListConfig) else metrics,
    )

    history = model.fit(
        x=train_data,
        validation_data=val_data,
        callbacks=[
            keras.callbacks.TensorBoard(log_dir=tensorboard_path),
            keras.callbacks.EarlyStopping(monitor="loss", patience=10, restore_best_weights=True),
            keras.callbacks.CSVLogger(filename=tensorboard_path / "train_log.csv"),
        ],
        class_weight={k: v / class_weight.weight_divisor for (k, v) in class_weight.weight_map.items()},
        verbose=2,
        **train_args,
    )

    return model, history


@hydra.main(config_path="../conf", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    seed = cfg.seed
    run_logdir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    model_cfg = cfg.model
    train_path = Path(cfg.data.dataset)

    if "mixed_precision" in cfg:
        tf.keras.mixed_precision.set_global_policy(cfg.mixed_precision)

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
        seed=cfg.seed,
        f_get_model=get_model,
        f_get_dataloader=get_dataloader,
        **cfg.train,
    )

    # TODO: add final validation plots here


if __name__ == "__main__":
    main()
