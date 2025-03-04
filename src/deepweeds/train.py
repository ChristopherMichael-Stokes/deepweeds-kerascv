import logging
from pathlib import Path
from typing import Callable, Dict, List, Tuple, cast

import hydra
import keras
import numpy as np
import tensorflow as tf
from omegaconf import DictConfig, ListConfig

from deepweeds.datasets.deep_weeds import get_train_val_dataloader
from deepweeds.models import not_resnet
from deepweeds.processing import PreProcessor
from deepweeds.utils import assert_type

log = logging.getLogger(__name__)


def train(
    focal_loss: bool,
    optimizer: str,
    optimizer_params: DictConfig,
    num_classes: int,
    metrics: List[str],
    batch_size: int,
    train_args: DictConfig,
    tensorboard_path: Path,
    model_cfg: DictConfig,
    augment: bool,
    seed: int,
    f_get_model: Callable[..., keras.Model],
    train_data: tf.data.Dataset,
    val_data: tf.data.Dataset,
    dtype_policy: str | None = "float32",
    loss_params: DictConfig | dict | None = None,
    early_stopping_params: DictConfig | None = None,
    lr_schedule_params: DictConfig | None = None,
    class_weight: DictConfig | None = None,
    save_model: bool | None = None,
) -> Tuple[keras.Model, Dict]:
    keras.utils.set_random_seed(seed)

    assert_type(train_data, tf.data.Dataset)
    assert_type(val_data, tf.data.Dataset)

    # A hack to fix the pre-processing pipeline which only works on float32 :(
    keras.mixed_precision.set_dtype_policy("float32")
    train_processor = PreProcessor(num_classes=num_classes, do_augment=augment)
    val_processor = PreProcessor(num_classes=num_classes, do_augment=False)
    log.info(f"Train processing pipeline: {train_processor}")
    log.info(f"Val processing pipeline: {val_processor}")

    train_data = (
        train_data.shuffle(
            buffer_size=batch_size * 4, seed=seed, reshuffle_each_iteration=True
        )
        .batch(batch_size, drop_remainder=True)
        .map(train_processor.preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_data = (
        val_data.batch(batch_size)
        .map(val_processor.preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    keras.mixed_precision.set_dtype_policy(dtype_policy)

    loss_params = dict(loss_params) if loss_params else {}

    if focal_loss:
        alpha = loss_params.get("alpha", 0.25)
        loss_params["alpha"] = (
            tf.convert_to_tensor(np.array(alpha))
            if not isinstance(alpha, float)
            else alpha
        )
        loss = keras.losses.CategoricalFocalCrossentropy(**loss_params)
    else:
        loss = keras.losses.CategoricalCrossentropy(**loss_params)

    model = f_get_model(**cast(dict, model_cfg))
    assert_type(model, keras.Model)
    model.compile(
        loss=loss,
        optimizer=getattr(keras.optimizers, optimizer)(**optimizer_params),
        metrics=list(metrics) if isinstance(metrics, ListConfig) else metrics,
    )

    callbacks = [
        keras.callbacks.TensorBoard(
            log_dir=tensorboard_path,
            write_steps_per_second=True,
        ),
        keras.callbacks.CSVLogger(filename=tensorboard_path / "train_log.csv"),
    ]

    if early_stopping_params:
        early_stopping = keras.callbacks.EarlyStopping(**early_stopping_params)
        callbacks.append(early_stopping)

    if lr_schedule_params:
        lr_schedule = keras.callbacks.ReduceLROnPlateau(**lr_schedule_params)
        callbacks.append(lr_schedule)

    if class_weight:
        divisor = (
            1 if "weight_divisor" not in class_weight else class_weight.weight_divisor
        )
        class_weight_map = {
            k: v / divisor for (k, v) in class_weight.weight_map.items()
        }
    else:
        class_weight_map = None

    history = model.fit(
        x=train_data,
        validation_data=val_data,
        callbacks=callbacks,
        class_weight=class_weight_map,
        verbose=2,
        **train_args,
    )

    if save_model:
        model.save(tensorboard_path / f"{model.name}.keras", zipped=True)

    return model, history


@hydra.main(config_path="../../conf", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    seed = cfg.seed
    run_logdir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    model_cfg = cfg.model
    train_path = Path(cfg.data.dataset)

    dtype_policy = None if "mixed_precision" not in cfg else cfg.mixed_precision

    train_dataloader, val_dataloader = get_train_val_dataloader(
        train_path=train_path,
        seed=seed,
        val_split=cfg.data.val_split,
    )

    if cfg.print_summary:
        if dtype_policy:
            keras.mixed_precision.set_global_policy(cfg.mixed_precision)
        model = not_resnet(**model_cfg)
        model.summary(print_fn=log.info)
        del model

    model, history = train(
        tensorboard_path=run_logdir,
        model_cfg=model_cfg,
        seed=cfg.seed,
        f_get_model=not_resnet,
        train_data=train_dataloader,
        val_data=val_dataloader,
        dtype_policy=dtype_policy,
        **cfg.train,
    )

    # TODO: add final validation plots here


if __name__ == "__main__":
    main()
