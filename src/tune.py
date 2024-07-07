import keras_tuner as kt

from datasets import deep_weeds
from models import not_resnet
from train import train


class HPoModel(kt.HyperModel):
    def build(self, hp: kt.HyperParameters): ...

    def fit(self, hp: kt.HyperParameters): ...
