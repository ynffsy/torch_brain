import torch
import torch.nn as nn


class Embedding(nn.Embedding):
    r"""A simple wrapper around `torch.nn.Embedding` with a custom initializer. 
    The embeddings are initialized with a normal distribution with mean 0 and
    standard deviation `init_scale`.
    
    By default, the `init_scale` is set to 0.02, which is the default value used in
    most transformer models.

    Refer to the documentation of `torch.nn.Embedding` for more details.
    """
    def __init__(
        self,
        *args,
        init_scale: float=0.02,
        **kwargs,
    ):
        self.init_scale = init_scale
        super().__init__(*args, **kwargs)

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.weight, mean=0, std=self.init_scale)
        self._fill_padding_idx_with_zero()
