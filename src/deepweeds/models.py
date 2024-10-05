import logging
from typing import Any, Callable, Dict, List, Tuple, cast

import keras
import keras_cv
import tensorflow as tf

log = logging.getLogger(__name__)


def not_resnet(
    input_shape: Tuple[int],
    n_classes: int,
    blocks: List[Dict] | None,
    output_activation: str,
    seed: int,
    block_dropout: float | None = None,
    scale_inputs: bool | None = None,
) -> keras.Model:

    MODEL_NAME = "MeNet"
    keras.utils.set_random_seed(seed)

    # Model layer definitions
    def conv_skip_block(
        x: tf.Tensor, filters: int, kernel_shape=(3, 3), padding="same", drop_rate=None
    ) -> tf.Tensor:
        """Implements a residual block basically the same as resnet - https://arxiv.org/pdf/1512.03385,
        however with spatial dropout for improved training stability."""
        skip = x
        x = keras.layers.Conv2D(filters, kernel_shape, padding=padding)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        if drop_rate:
            x = keras.layers.SpatialDropout2D(drop_rate)(x)
        x = keras.layers.Conv2D(filters, kernel_shape, padding=padding)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Concatenate()([x, skip])
        x = keras.layers.ReLU()(x)
        return x

    input_ = keras.Input(shape=input_shape)
    scaling = keras_cv.layers.Rescaling(scale=1.0 / 255)
    input_layer = keras.layers.Conv2D(64, (7, 7), (2, 2), padding="same")
    pool_layer = keras.layers.MaxPool2D(pool_size=(2, 2))
    global_avg_pool = keras.layers.GlobalAveragePooling2D()
    output_bn = keras.layers.BatchNormalization()
    output_layer = keras.layers.Dense(n_classes, activation=output_activation)

    # Model data flow construction
    # Input conv + pool
    x = input_layer(input_)
    if scale_inputs:
        x = scaling(x)
    x = pool_layer(x)

    # Residual blocks
    for block in blocks or []:
        x = conv_skip_block(x, drop_rate=block_dropout, **block)

    # Output pool + fully connected
    x = global_avg_pool(x)
    x = output_bn(x)
    output = output_layer(x)

    return keras.Model(name=MODEL_NAME, inputs=[input_], outputs=[output])
