import keras_tuner as kt


class HPoModel(kt.HyperModel):
    def build(self, hp: kt.HyperParameters): ...

    def fit(self, hp: kt.HyperParameters): ...
