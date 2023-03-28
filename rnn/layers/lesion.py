from purias_utils.rnn.augmentation.lesion import *
from purias_utils.rnn.layers.base import WeightLayer

class LesionWeightLayer(WeightLayer):

    @property
    def masked_weight(self):
        raise NotImplementedError

