from .random_time_scaling import RandomTimeScaling
from .random_crop import RandomCrop
from .unit_dropout import UnitDropout
from .output_sampler import RandomOutputSampler


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data