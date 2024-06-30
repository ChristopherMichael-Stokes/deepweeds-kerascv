from pathlib import Path
from typing import Tuple

import tensorflow as tf


def get_train_val_dataloader(
    train_path: Path,
    batch_size: int | None = None,
    seed=42,
    val_split=0.1,
    verbose=False,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    class_names = [
        "rubber_vine",
        "negative",
        "parthenium",
        "chinee_apple",
        "prickly_acacia",
        "snake_weed",
        "parkinsonia",
        "siam_weed",
        "lantana",
    ]

    train_data, val_data = tf.keras.preprocessing.image_dataset_from_directory(
        train_path,
        class_names=class_names,
        image_size=(256, 256),
        validation_split=val_split,  # Makes the split at the last N% of files, so with shuffling is just a random split
        subset="both",  # Loads validation and train splits
        batch_size=batch_size,
        interpolation="nearest",
        crop_to_aspect_ratio=True,
        seed=seed,
        shuffle=True,
        verbose=verbose,
    )

    return train_data, val_data
