from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType

from kirby.nn import RotaryEmbedding, RotaryCrossAttention, RotarySelfAttention


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(p=dropout),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x):
        return self.net(x)
    

class PerceiverRotary(nn.Module):
    def __init__(
        self,
        *,
        dim=512,
        context_dim=None,
        dim_head=64,
        depth=2,
        cross_heads=1,
        self_heads=8,
        ffn_dropout=0.2,
        lin_dropout=0.4,
        atn_dropout=0.0,
        use_memory_efficient_attn=True,
    ):
        super().__init__()

        self.rotary_emb = RotaryEmbedding(dim_head)

        self.dropout = nn.Dropout(p=lin_dropout)

        # Encoding transformer (q-latent, kv-input spikes)
        self.enc_atn = RotaryCrossAttention(
            dim=dim,
            context_dim=context_dim,
            heads=cross_heads,
            dropout=atn_dropout,
            dim_head=dim_head,
            rotate_value=True,
        )
        self.enc_ffn = nn.Sequential(
            nn.LayerNorm(dim), FeedForward(dim=dim, dropout=ffn_dropout)
        )

        # Processing transfomers (qkv-latent)
        self.proc_layers = nn.ModuleList([])
        for i in range(depth):
            self.proc_layers.append(
                nn.ModuleList(
                    [
                        RotarySelfAttention(
                            dim=dim,
                            heads=self_heads,
                            dropout=atn_dropout,
                            dim_head=dim_head,
                            use_memory_efficient_attn=use_memory_efficient_attn,
                            rotate_value=True,
                        ),
                        nn.Sequential(
                            nn.LayerNorm(dim),
                            FeedForward(dim=dim, dropout=ffn_dropout),
                        ),
                    ]
                )
            )

        self.dec_atn = RotaryCrossAttention(
            dim=dim,
            heads=cross_heads,
            dropout=atn_dropout,
            dim_head=dim_head,
            rotate_value=False,
        )
        self.dec_ffn = nn.Sequential(
            nn.LayerNorm(dim), FeedForward(dim=dim, dropout=ffn_dropout)
        )

        self.dim = dim

    def forward(
        self,
        *,
        inputs, # (B, N_in, dim)
        latents, # (B, N_latent, dim)
        outputs=None, # (B, N_out, dim)
        input_timestamps, # (B, N_in)
        latent_timestamps, # (B, N_latent)
        output_timestamps=None, # (B, N_out)
        input_mask=None, # (B, N_in)
    ) -> Tuple[
        TensorType["batch", "*nqueries", "*dim"]
    ]:
        # compute timestamp embeddings
        input_timestamp_emb = self.rotary_emb(input_timestamps)
        latent_timestamp_emb = self.rotary_emb(latent_timestamps)
        output_timestamp_emb = self.rotary_emb(output_timestamps)

        # encode
        latents = latents + self.enc_atn(
            latents,
            inputs,
            latent_timestamp_emb,
            input_timestamp_emb,
            input_mask,
        )
        latents = latents + self.enc_ffn(latents)

        # process
        for self_attn, self_ff in self.proc_layers:
            latents = latents + self.dropout(self_attn(latents, latent_timestamp_emb))
            latents = latents + self.dropout(self_ff(latents))

        if outputs is None:
            return latents

        # decode
        outputs = outputs + self.dec_atn(
            outputs, latents, output_timestamp_emb, latent_timestamp_emb
        )
        outputs = outputs + self.dec_ffn(outputs)

        return outputs
