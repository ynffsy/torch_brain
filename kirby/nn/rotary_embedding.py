import torch
import torch.nn as nn
from einops import repeat, rearrange


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, t_min=1e-4, t_max=4.0):
        super().__init__()
        inv_freq = torch.zeros(dim // 2)
        inv_freq[: dim // 4] = (
            2
            * torch.pi
            / (
                t_min
                * (
                    (t_max / t_min)
                    ** (torch.arange(0, dim // 2, 2).float() / (dim // 2))
                )
            )
        )

        self.register_buffer("inv_freq", inv_freq)

    def forward(self, timestamps):
        freqs = torch.einsum("..., f -> ... f", timestamps, self.inv_freq)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        return freqs


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


def apply_rotary_pos_emb(freqs, x, dim=2):
    if dim == 1:
        freqs = rearrange(freqs, "n ... -> n () ...")
    elif dim == 2:
        freqs = rearrange(freqs, "n m ... -> n m () ...")
    x = (x * freqs.cos()) + (rotate_half(x) * freqs.sin())
    return x
