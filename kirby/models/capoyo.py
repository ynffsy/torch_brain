from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torchtyping import TensorType
from einops import rearrange, repeat

from kirby.taxonomy import DecoderSpec, Decoder, Cre_line
from kirby.nn import (
    Embedding,
    InfiniteVocabEmbedding,
    MultitaskReadout,
    PerceiverRotary,
    prepare_for_multitask_readout,
)
from kirby.data import pad, chain, track_mask, track_batch
from kirby.utils import (
    create_start_end_unit_tokens,
    create_linspace_latent_tokens,
    get_sinusoidal_encoding,
)

from kirby.models.poyo_plus import BACKEND_CONFIGS


class CaPOYO(nn.Module):
    def __init__(
        self,
        *,
        dim=512,
        dim_head=64,
        num_latents=64,
        patch_size=1,
        depth=2,
        cross_heads=1,
        self_heads=8,
        ffn_dropout=0.2,
        lin_dropout=0.4,
        atn_dropout=0.0,
        emb_init_scale=0.02,
        use_cre_line_embedding=False,
        backend_config="gpu_fp32",
        decoder_specs: Dict[str, DecoderSpec],
    ):
        super().__init__()

        self.dim = dim
        self.patch_size = patch_size

        # input embs
        self.unit_emb = InfiniteVocabEmbedding(dim, init_scale=emb_init_scale)
        self.token_type_emb = Embedding(4, dim, init_scale=emb_init_scale)
        self.value_embedding_layer = nn.Linear(patch_size, dim, bias=False)
        self.unit_feat_embedding_layer = nn.Linear(3, dim, bias=True)
        self.use_cre_line_embedding = use_cre_line_embedding
        if self.use_cre_line_embedding:
            self.cre_line_embedding_layer = Embedding(
                Cre_line.max_value() + 1, dim, init_scale=emb_init_scale
            )

        # latent embs
        self.latent_emb = Embedding(num_latents, dim, init_scale=emb_init_scale)

        # output embs
        self.session_emb = InfiniteVocabEmbedding(dim, init_scale=emb_init_scale)
        self.task_emb = Embedding(
            Decoder.max_value() + 1, dim, init_scale=emb_init_scale
        )

        # determine backend
        if backend_config not in BACKEND_CONFIGS.keys():
            raise ValueError(
                f"Invalid backend config: {backend_config}, must be one of"
                f" {list(BACKEND_CONFIGS.keys())}"
            )

        self.batch_type = BACKEND_CONFIGS[backend_config][0]

        context_dim = 4 * dim if not self.use_cre_line_embedding else 5 * dim

        self.perceiver_io = PerceiverRotary(
            dim=dim,
            context_dim=context_dim,
            dim_head=dim_head,
            depth=depth,
            cross_heads=cross_heads,
            self_heads=self_heads,
            ffn_dropout=ffn_dropout,
            lin_dropout=lin_dropout,
            atn_dropout=atn_dropout,
            backend=BACKEND_CONFIGS[backend_config][1],
        )

        # Output projections + loss
        self.readout = MultitaskReadout(
            latent_dim=dim,
            decoder_specs=decoder_specs,
            batch_type=self.batch_type[2],
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
        *,
        # input sequence
        unit_index,  # (B, N_in)
        timestamps,  # (B, N_in)
        patches,  # (B, N_in, N_feats)
        token_type,  # (B, N_in)
        unit_feats,  # (B, N_in, N_feats)
        unit_spatial_emb,  # (B, N_in, dim)
        unit_cre_line=None,  # (B, N_in)
        input_mask=None,  # (B, N_in)
        input_seqlen=None,
        # latent sequence
        latent_index,  # (B, N_latent)
        latent_timestamps,  # (B, N_latent)
        latent_seqlen=None,
        # output sequence
        session_index,  # (B,)
        output_timestamps,  # (B, N_out)
        output_decoder_index,  # (B, N_out)
        output_seqlen=None,
        output_batch_index=None,
        output_values: Optional[Dict[str, torch.Tensor]] = None,
        output_weights: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[
        Dict[str, TensorType["batch", "*nqueries", "*nchannelsout"]],
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        # input
        input_feats = [
            self.unit_emb(unit_index) + self.token_type_emb(token_type),
            self.value_embedding_layer(patches),
            self.unit_feat_embedding_layer(unit_feats),
            unit_spatial_emb,
        ]
        if self.use_cre_line_embedding:
            input_feats.append(self.cre_line_embedding_layer(unit_cre_line))
        inputs = torch.cat(
            input_feats,
            dim=-1,
        )

        # latents
        latents = self.latent_emb(latent_index)

        # outputs
        output_queries = self.task_emb(output_decoder_index) + self.session_emb(
            session_index
        )

        # feed into perceiver
        output_latents = self.perceiver_io(
            inputs=inputs,
            latents=latents,
            output_queries=output_queries,
            input_timestamps=timestamps,
            latent_timestamps=latent_timestamps,
            output_query_timestamps=output_timestamps,
            input_mask=input_mask,
            input_seqlen=input_seqlen,
            latent_seqlen=latent_seqlen,
            output_query_seqlen=output_seqlen,
        )

        # Readout layer
        output, loss, losses_taskwise = self.readout(
            output_latents=output_latents,
            output_decoder_index=output_decoder_index,
            output_batch_index=output_batch_index,
            output_values=output_values,
            output_weights=output_weights,
        )

        return output, loss, losses_taskwise


class CaPOYOTokenizer:
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
        latent_step,
        num_latents_per_step,
        dim,
        patch_size,
        batch_type,
        eval=False,
        use_cre_line_embedding=False,
    ):
        self.unit_tokenizer = unit_tokenizer
        self.session_tokenizer = session_tokenizer

        self.decoder_registry = decoder_registry

        self.latent_step = latent_step
        self.num_latents_per_step = num_latents_per_step
        self.dim = dim
        self.patch_size = patch_size

        self.batch_type = batch_type
        self.eval = eval

        self.use_cre_line_embedding = use_cre_line_embedding

    def __call__(self, data):
        # context window
        start, end = 0.0, 1.0

        ### prepare input
        unit_ids = data.units.id
        unit_lvl_spatial_emb = get_sinusoidal_encoding(
            data.units.imaging_plane_xy[:, 0],
            data.units.imaging_plane_xy[:, 1],
            self.dim // 2,
        ).astype(np.float32)
        unit_lvl_feats = np.stack(
            [
                data.units.imaging_plane_area,
                data.units.imaging_plane_width,
                data.units.imaging_plane_height,
            ],
            axis=1,
        ).astype(np.float32)

        calcium_traces = data.calcium_traces.df_over_f.astype(
            np.float32
        )  # (time x num_rois)
        timestamps = data.calcium_traces.timestamps.astype(np.float32)
        num_rois = calcium_traces.shape[1]

        # patch tokenization
        # clip the time dimension to accomodate the patch size
        # WARNING: it is important to still have a multiple of patch_size
        # this is a fix to deal with the arbitrary slicing that might happen
        num_frames = calcium_traces.shape[0] // self.patch_size * self.patch_size
        if num_frames == 0:
            raise ValueError(
                f"The patch size ({self.patch_size}) is larger than "
                f"sequence length ({calcium_traces.shape[0]})."
            )
        calcium_traces = calcium_traces[:num_frames]
        timestamps = timestamps[:num_frames]

        calcium_traces = calcium_traces.reshape(
            -1, self.patch_size, calcium_traces.shape[1]
        )
        timestamps = timestamps.reshape(-1, self.patch_size).mean(axis=1)

        # now flatten
        patches = rearrange(calcium_traces, "t d c -> (t c) d")
        unit_index = repeat(np.arange(num_rois), "c -> (t c)", t=timestamps.shape[0])
        unit_feats = repeat(unit_lvl_feats, "c f -> (t c) f", t=timestamps.shape[0])
        unit_spatial_emb = repeat(
            unit_lvl_spatial_emb, "c d -> (t c) d", t=timestamps.shape[0]
        )
        timestamps = repeat(timestamps, "t -> (t c)", c=num_rois)

        # create start and end tokens for each unit
        (
            se_token_type_index,
            se_unit_index,
            se_timestamps,
        ) = create_start_end_unit_tokens(unit_ids, start, end)

        # append start and end tokens to the spike sequence
        token_type_index = np.concatenate(
            [se_token_type_index, np.zeros_like(unit_index)]
        )
        unit_index = np.concatenate([se_unit_index, unit_index])
        timestamps = np.concatenate([se_timestamps, timestamps])
        patches = np.concatenate(
            [
                np.zeros((se_unit_index.shape[0], patches.shape[1]), dtype=np.float32),
                patches,
            ]
        )
        unit_feats = np.concatenate(
            [
                unit_lvl_feats[se_unit_index],
                unit_feats,
            ]
        )
        unit_spatial_emb = np.concatenate(
            [
                unit_lvl_spatial_emb[se_unit_index],
                unit_spatial_emb,
            ]
        )

        # unit_index is relative to the recording, so we want it to map it to
        # the global unit index
        local_to_global_map = np.array(self.unit_tokenizer(unit_ids))
        unit_index = local_to_global_map[unit_index]

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
            output_subtask_index,
        ) = prepare_for_multitask_readout(
            data,
            self.decoder_registry,
        )

        if self.use_cre_line_embedding:
            subject_cre_line = data.subject.cre_line
            subject_cre_line_index = Cre_line.from_string(subject_cre_line).value
            unit_cre_line = np.full_like(unit_index, subject_cre_line_index)

        batch = {}
        if self.batch_type[0] == "stacked":
            batch = {
                **batch,
                # input sequence
                "unit_index": pad(unit_index),
                "timestamps": pad(timestamps),
                "patches": pad(patches),
                "unit_feats": pad(unit_feats),
                "token_type": pad(token_type_index),
                "unit_spatial_emb": pad(unit_spatial_emb),
                "input_mask": track_mask(unit_index),
                # latent sequence
                "latent_index": latent_index,
                "latent_timestamps": latent_timestamps,
            }
            if self.use_cre_line_embedding:
                batch["unit_cre_line"] = pad(unit_cre_line)
        else:
            batch = {
                **batch,
                # input sequence
                "unit_index": chain(unit_index),
                "timestamps": chain(timestamps),
                "patches": chain(patches),
                "unit_feats": chain(unit_feats),
                "token_type": chain(token_type_index),
                "unit_spatial_emb": chain(unit_spatial_emb),
                "input_seqlen": len(unit_index),
                # latent sequence
                "latent_index": chain(latent_index),
                "latent_timestamps": chain(latent_timestamps),
                "latent_seqlen": len(latent_index),
            }
            if self.use_cre_line_embedding:
                batch["unit_cre_line"] = chain(unit_cre_line)
        if self.batch_type[1] == "chained":
            batch["latent_seqlen"] = len(latent_index)

        if self.batch_type[2] == "stacked":
            batch = {
                **batch,
                # output sequence
                "session_index": pad(np.repeat(session_index, len(output_timestamps))),
                "output_timestamps": pad(output_timestamps),
                "output_decoder_index": pad(output_task_index),
                "output_values": chain(output_values),
                "output_weights": chain(output_weights),
            }
        else:
            batch = {
                **batch,
                # output sequence
                "session_index": chain(session_index),
                "output_timestamps": chain(output_timestamps),
                "output_decoder_index": chain(output_task_index),
                "output_seqlen": len(output_timestamps),
                "output_batch_index": track_batch(output_timestamps),
                "output_values": chain(output_values),
                "output_weights": chain(output_weights),
            }

        if self.eval:
            # we will add a few more fields needed for evaluation
            batch["session_id"] = data.session
            batch["absolute_start"] = data.absolute_start
            batch["output_subtask_index"] = chain(output_subtask_index)

        return batch
