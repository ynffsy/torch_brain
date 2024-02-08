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
    prepare_for_multitask_readout,
    extract_request_keys_from_decoder_registry,
)
from kirby.data import pad, chain, track_mask
from kirby.utils import (
    create_start_end_unit_tokens,
    create_linspace_latent_tokens,
    inspect_request_keys,
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


class POYOTokenizer:
    r"""Tokenizer used to tokenize Data for the POYO1 model.

    This tokenizer can be called as a transform. If you are applying multiple
    transforms, make sure to apply this one last.

    Args:
        unit_tokenizer (Callable): Tokenizer for the units.
        session_tokenizer (Callable): Tokenizer for the sessions.
        decoder_registry (Dict): Registry of the decoders.
        weight_registry (Dict): Registry of the weights.
        latent_step (float): Step size for generating latent tokens.
        num_latents_per_step (int): Number of latents per step.
    """

    def __init__(
        self,
        unit_tokenizer,
        session_tokenizer,
        decoder_registry,
        weight_registry,
        latent_step,
        num_latents_per_step,
    ):
        self.unit_tokenizer = unit_tokenizer
        self.session_tokenizer = session_tokenizer

        self.decoder_registry = decoder_registry
        self.weight_registry = weight_registry

        self.latent_step = latent_step
        self.num_latents_per_step = num_latents_per_step

    @property
    def request_keys(self):
        return inspect_request_keys(self.__call__) + extract_request_keys_from_decoder_registry(self.decoder_registry)

    def __call__(self, data):
        # context window
        start, end = data.start, data.end

        ### prepare input
        unit_ids = data.units.id
        spike_unit_index = data.spikes.unit_index
        spike_timestamps = data.spikes.timestamps

        # create start and end tokens for each unit
        (
            se_token_type_index,
            se_unit_index,
            se_timestamps,
        ) = create_start_end_unit_tokens(unit_ids, start, end)

        # append start and end tokens to the spike sequence
        spike_token_type_index = np.concatenate(
            [se_token_type_index, np.zeros_like(spike_unit_index)]
        )
        spike_unit_index = np.concatenate([se_unit_index, spike_unit_index])
        spike_timestamps = np.concatenate([se_timestamps, spike_timestamps])

        # unit_index is relative to the recording, so we want it to map it to
        # the global unit index
        local_to_global_map = np.array(self.unit_tokenizer(unit_ids))
        spike_unit_index = local_to_global_map[spike_unit_index]

        ### prepare latents
        latent_index, latent_timestamps = create_linspace_latent_tokens(
            start,
            end,
            step=self.latent_step,
            num_latents_per_step=self.num_latents_per_step,
        )

        ### prepare outputs
        session_index = self.session_tokenizer(data.session)

        (
            output_timestamps,
            output_task_index,
            output_values,
            output_weights,
        ) = prepare_for_multitask_readout(
            data, self.decoder_registry, self.weight_registry
        )

        return {
            "spike_unit_index": pad(spike_unit_index),
            "spike_timestamps": pad(spike_timestamps),
            "spike_type": pad(spike_token_type_index),
            "input_mask": track_mask(spike_unit_index),
            "latent_index": latent_index,
            "latent_timestamps": latent_timestamps,
            "session_index": session_index,
            "output_timestamps": pad(output_timestamps),
            "output_task_index": pad(output_task_index),
            "output_values": chain(output_values),
            "output_weights": chain(output_weights),
        }
