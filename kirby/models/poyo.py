from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torchtyping import TensorType

from kirby.taxonomy import DecoderSpec, Output
from kirby.nn import (
    Embedding,
    InfiniteVocabEmbedding,
    MultitaskReadout,
    PerceiverRotary,
)


class POYO(nn.Module):
    def __init__(
        self,
        *,
        dim=512,
        dim_head=64,
        num_latents=64,
        depth=2,
        cross_heads=1,
        self_heads=8,
        ffn_dropout=0.2,
        lin_dropout=0.4,
        atn_dropout=0.0,
        emb_init_scale=0.02,
        use_memory_efficient_attn=True,
        task_specs: Dict[str, DecoderSpec],
    ):
        super().__init__()

        self.unit_emb = InfiniteVocabEmbedding(dim, init_scale=emb_init_scale)
        self.session_emb = InfiniteVocabEmbedding(dim, init_scale=emb_init_scale)
        self.spike_type_emb = Embedding(4, dim, init_scale=emb_init_scale)
        self.task_emb = Embedding(Output.max_value(), dim, init_scale=emb_init_scale)
        self.latent_emb = Embedding(num_latents, dim, init_scale=emb_init_scale)

        self.perceiver_io = PerceiverRotary(
            dim=dim,
            dim_head=dim_head,
            depth=depth,
            cross_heads=cross_heads,
            self_heads=self_heads,
            ffn_dropout=ffn_dropout,
            lin_dropout=lin_dropout,
            atn_dropout=atn_dropout,
            use_memory_efficient_attn=use_memory_efficient_attn,
        )

        # Output projections + loss
        self.readout = MultitaskReadout(
            latent_dim=dim,
            task_specs=task_specs,
        )

        self.dim = dim

    def freeze_middle(self) -> List[nn.Module]:
        # Freeze everything except the readout, unit embedding, and session embedding
        # layers.
        middle_modules = []
        banned_modules = [
            self.readout,
            self.unit_emb,
            self.session_emb,
            self.enc_atn,
            self.enc_ffn,
        ]
        for module in self.children():
            if module in banned_modules:
                continue
            for param in module.parameters():
                param.requires_grad = False
            middle_modules.append(module)

        return middle_modules

    def unfreeze_middle(self) -> None:
        for module in self.children():
            for param in module.parameters():
                param.requires_grad = True

    def forward(
        self,
        # input sequence
        spike_unit_index,  # (B, N_in)
        spike_timestamps,  # (B, N_in)
        spike_type,  # (B, N_in)
        input_mask,  # (B, N_in)
        # latent sequence
        latent_index,  # (B, N_latent)
        latent_timestamps,  # (B, N_latent)
        # output sequence
        session_index,  # (B,)
        output_timestamps,  # (B, N_out)
        output_task_index,  # (B, N_out)
        output_values: Optional[Dict[str, torch.Tensor]] = None,
        output_weights: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[
        Dict[str, TensorType["batch", "*nqueries", "*nchannelsout"]],
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        # input
        inputs = self.unit_emb(spike_unit_index) + self.spike_type_emb(spike_type)

        # latents
        latents = self.latent_emb(latent_index)

        # outputs
        outputs = self.task_emb(output_task_index) + self.session_emb(
            session_index
        ).unsqueeze(1)

        # feed into perceiver
        output_latents = self.perceiver_io(
            inputs=inputs,
            latents=latents,
            outputs=outputs,
            input_timestamps=spike_timestamps,
            latent_timestamps=latent_timestamps,
            output_timestamps=output_timestamps,
            input_mask=input_mask,
        )

        # Readout layer
        output, loss, losses_taskwise = self.readout(
            output_latents=output_latents,
            output_task_index=output_task_index,
            output_values=output_values,
            output_weights=output_weights,
        )

        return output, loss, losses_taskwise
