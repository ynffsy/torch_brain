import torch


class RandomTimeScaling:
    def __init__(self, min_scale, max_scale):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, data):
        scale = torch.rand(1).item() * (self.max_scale - self.min_scale) + self.min_scale
        data.start *= scale
        data.end *= scale

        # Note, this only works properly for classification outputs, since outputs are 
        # not rescaled.
        data.spikes.timestamps *= scale

        if hasattr(data, 'lfps'):
            data.lfps.timestamps *= scale

        return data
