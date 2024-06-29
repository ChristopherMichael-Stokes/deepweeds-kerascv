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

from dataset import get_train_val_dataloader

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def get_run_logdir(root_logdir) -> Path:
    root_logir_path = Path(root_logdir)
    root_logir_path.mkdir(parents=True, exist_ok=True)
    return root_logir_path / strftime("run_%Y_%m_%d_%H_%M_%S")


def get_model(
    seed: int,
    input_shape: Tuple[int],
    n_classes: int,
    hidden_layers: List[Dict],
    output_activation: str,
) -> keras.Model:
    MODEL_NAME = "MeNet"

    tf.random.set_seed(seed)

    # TODO: implement reading layer setup from config
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=input_shape),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(n_classes, activation="softmax"),
        ],
        name=MODEL_NAME,
    )

    return model


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
    f_get_model: Callable[Any, keras.Model],
    f_get_dataloader: Callable[Any, tf.data.Dataset],
) -> Dict:

    train_data, val_data = f_get_dataloader()
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

    model = f_get_model(**model_cfg)
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
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        ],
        class_weight={k: v / class_weight.weight_divisor for (k, v) in class_weight.weight_map.items()},
        **train_args,
    )

    return history


@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    seed = cfg.seed
    logdir = cfg.output.logdir
    model_cfg = cfg.model
    train_path = Path(cfg.data.dataset)

    get_dataloader = partial(
        get_train_val_dataloader, train_path=train_path, seed=seed, val_split=cfg.data.val_split
    )

    run_logdir = get_run_logdir(logdir)

    if cfg.print_summary:
        model = get_model(**model_cfg)
        print(model.summary())
        del model

    train(
        tensorboard_path=run_logdir,
        model_cfg=model_cfg,
        f_get_model=get_model,
        f_get_dataloader=get_dataloader,
        **cfg.train,
    )


if __name__ == "__main__":
    main()
