import logging
from typing import Tuple

import keras_cv
import tensorflow as tf

log = logging.getLogger(__name__)


class PreProcessor:
    def __init__(self, num_classes: int, do_augment=False, augmentations_per_image=1):
        # TODO: add simple augmentations + visualise outputs in a notebook,
        # One hunch is that a saturation augment would help as there seems to be
        # a lot of variation between image samples or image locations

        # jitter = keras_cv.layers.RandomColorJitter(
        #     value_range=(0, 255),
        #     brightness_factor=(-0.2, 0.2),
        #     contrast_factor=(0.5, 0.7),
        #     saturation_factor=(0.4, 0.6),
        #     hue_factor=(0.0, 0.3),
        # )
        self.num_classes = num_classes
        self.do_augment = do_augment
        self.augmentation_pipeline = keras_cv.layers.RandomAugmentationPipeline(
            [
                keras_cv.layers.RandomFlip(mode="horizontal"),
                keras_cv.layers.RandomChannelShift(value_range=(0, 255), factor=0.4),
                keras_cv.layers.RandomColorDegeneration(factor=0.9),
                keras_cv.layers.RandomCutout(0.5, 0.5),
            ],
            augmentations_per_image=augmentations_per_image,
        )
        self.cut_mix = keras_cv.layers.CutMix()
        self.mix_up = keras_cv.layers.MixUp()

    def cut_mix_and_mix_up(self, samples):
        samples = self.cut_mix(samples, training=True)
        samples = self.mix_up(samples, training=True)
        return samples

    def preprocess_data(
        self,
        images: tf.Tensor,
        labels: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        labels = tf.one_hot(labels, self.num_classes)
        inputs = {"images": images, "labels": labels}

        outputs = inputs

        if self.do_augment:
            outputs = self.augmentation_pipeline(outputs)
            # outputs = self.cut_mix_and_mix_up(outputs)

        return outputs["images"], outputs["labels"]

    def __repr__(self):
        return str(
            f"{self.__class__.__name__}, do_augment={self.do_augment},"
            f" augmentation_pipeline={self.augmentation_pipeline}"
        )
