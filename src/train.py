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
from models import not_resnet
from processing import PreProcessor

log = logging.getLogger(__name__)


def train(
    loss: str,
    optimizer: str,
    optimizer_params: DictConfig,
    num_classes: int,
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

    train_processor = PreProcessor(num_classes=num_classes, do_augment=augment)
    val_processor = PreProcessor(num_classes=num_classes, do_augment=False)
    log.info(f"Train processing pipeline: {train_processor}")
    log.info(f"Val processing pipeline: {val_processor}")

    train_data = (
        train_data.shuffle(buffer_size=batch_size * 4, seed=seed, reshuffle_each_iteration=True)
        .batch(batch_size, drop_remainder=True)
        .map(train_processor.preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_data = (
        val_data.batch(batch_size)
        .map(val_processor.preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    f_loss = getattr(tf.keras.losses, loss)
    if loss_params:
        f_loss = partial(f_loss, **loss_params)  # type: ignore

    model = f_get_model(**model_cfg)  # type: ignore
    assert isinstance(model, keras.Model)
    model.compile(
        loss=f_loss,
        optimizer=getattr(tf.keras.optimizers, optimizer)(**optimizer_params),
        metrics=list(metrics) if isinstance(metrics, ListConfig) else metrics,
    )

    lr_schedule = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
    )

    log.info(f"Learning rate schedule: {lr_schedule}")
    history = model.fit(
        x=train_data,
        validation_data=val_data,
        callbacks=[
            keras.callbacks.TensorBoard(log_dir=tensorboard_path),
            keras.callbacks.EarlyStopping(monitor="loss", patience=10, restore_best_weights=True),
            keras.callbacks.CSVLogger(filename=tensorboard_path / "train_log.csv"),
            lr_schedule,
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
        model = not_resnet(**model_cfg)
        model.summary(print_fn=log.info)
        del model

    model, history = train(
        tensorboard_path=run_logdir,
        model_cfg=model_cfg,
        seed=cfg.seed,
        f_get_model=not_resnet,
        f_get_dataloader=get_dataloader,
        **cfg.train,
    )

    # TODO: add final validation plots here


if __name__ == "__main__":
    main()
