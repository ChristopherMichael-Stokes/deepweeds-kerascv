import logging
from pathlib import Path
from time import strftime
from typing import Any, Dict, List, Tuple, cast

import hydra
import keras
import keras_tuner as kt
import numpy as np
import tensorflow as tf
from omegaconf import DictConfig, ListConfig

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def load_data(dataset: str) -> tf.data.Dataset:
    return getattr(tf.keras.datasets, dataset).load_data()


def get_run_logdir(root_logdir) -> Path:
    root_logir_path = Path(root_logdir)
    root_logir_path.mkdir(parents=True, exist_ok=True)
    return root_logir_path / strftime("run_%Y_%m_%d_%H_%M_%S")


def get_model(
    seed: int,
    n_classes: int,
    hidden_layers: List[Dict],
    output_activation: str,
    X_train: np.ndarray,
) -> keras.Model:
    MODEL_NAME = "WideAndDeep"

    tf.random.set_seed(seed)
    normalization_layer = tf.keras.layers.Normalization()
    input_ = tf.keras.layers.Input(shape=X_train.shape[1:])
    flatten = tf.keras.layers.Flatten()

    hidden_list: List[keras.Layer] = []
    for layer in hidden_layers:
        hidden_list.append(tf.keras.layers.Dense(layer["n_units"], activation=layer["activation"]))

    concat_layer = tf.keras.layers.Concatenate()
    output_layer = tf.keras.layers.Dense(n_classes, activation=output_activation)

    normalized = normalization_layer(flatten(input_))
    hidden = normalized
    for layer in hidden_list:
        hidden = layer(hidden)
    concat = concat_layer([normalized, hidden])
    output = output_layer(concat)

    normalization_layer.adapt(X_train.reshape(X_train.shape[0], -1))

    return tf.keras.Model(name=MODEL_NAME, inputs=[input_], outputs=[output])


def train(
    model: keras.Model,
    loss: str,
    optimizer: str,
    optimizer_params: DictConfig,
    metrics: List[str],
    train_args: DictConfig,
    tensorboard_path: Path,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
) -> Dict:
    model.compile(
        loss=loss,
        optimizer=getattr(tf.keras.optimizers, optimizer)(**optimizer_params),
        metrics=list(metrics) if isinstance(metrics, ListConfig) else metrics,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_valid, y_valid),
        callbacks=[
            keras.callbacks.TensorBoard(log_dir=tensorboard_path),
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        ],
        **train_args,
    )

    return history


@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    dataset = cfg.data.dataset
    logdir = cfg.output.logdir
    model_cfg = cfg.model

    (X_train_full, y_train_full), (X_test, y_test) = load_data(dataset)
    X_train, y_train = X_train_full[: -cfg.data.val_samples], y_train_full[: -cfg.data.val_samples]
    X_valid, y_valid = X_train_full[-cfg.data.val_samples :], y_train_full[-cfg.data.val_samples :]

    run_logdir = get_run_logdir(logdir)
    model = get_model(**model_cfg, X_train=X_train)

    if cfg.print_summary:
        print(model.summary())

    train(
        model=model,
        tensorboard_path=run_logdir,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        **cfg.train,
    )


if __name__ == "__main__":
    main()
