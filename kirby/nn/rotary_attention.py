import logging

import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange, repeat

try:
    import xformers.ops as xops
except ImportError:
    xops = None


from kirby.nn.rotary_embedding import apply_rotary_pos_emb


class RotaryCrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        context_dim: int = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        rotate_value: bool = False,
    ):
        super().__init__()

        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else dim
        self.scale = dim_head**-0.5
        self.heads = heads
        self.dropout = dropout
        self.rotate_value = rotate_value

        # build networks
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(
        self,
        x_query,
        x_context,
        rotary_pos_emb_query,
        rotary_pos_emb_context,
        attn_mask=None,
    ):
        # normalize
        x_query = self.norm(x_query)
        x_context = self.norm_context(x_context)

        # calculate query, key, value
        q = self.to_q(x_query)
        k, v = self.to_kv(x_context).chunk(2, dim=-1)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)

        # apply rotary embeddings
        q = apply_rotary_pos_emb(rotary_pos_emb_query, q, dim=1)
        k = apply_rotary_pos_emb(rotary_pos_emb_context, k, dim=1)
        if self.rotate_value:
            v = apply_rotary_pos_emb(rotary_pos_emb_context, v, dim=1)

        # attention mask
        attn_mask = (
            rearrange(attn_mask, "b n -> b () () n") if attn_mask is not None else None
        )
        # perform attention, by default will use the optimal attention implementation
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout
        )

        if self.rotate_value:
            out = apply_rotary_pos_emb(-rotary_pos_emb_query, out, dim=1)

        # project back to output
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class RotarySelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        use_memory_efficient_attn=True,
        rotate_value=False,
    ):
        super().__init__()

        if use_memory_efficient_attn and xops is None:
            logging.warning(
                "xformers is not installed, falling back to default attention"
            )
            use_memory_efficient_attn = False

        inner_dim = dim_head * heads
        self.scale = dim_head**-0.5
        self.heads = heads
        self.dropout = dropout
        self.use_memory_efficient_attn = use_memory_efficient_attn
        self.rotate_value = rotate_value

        # build networks
        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, rotary_pos_emb, attn_mask=None):
        # normalize
        x = self.norm(x)

        # calculate query, key, value
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        if self.use_memory_efficient_attn:
            # xformers attention expects shape B, N, H, D instead of B, H, N, D
            q = rearrange(q, "b n (h d) -> b n h d", h=self.heads)
            k = rearrange(k, "b n (h d) -> b n h d", h=self.heads)
            v = rearrange(v, "b n (h d) -> b n h d", h=self.heads)

            # apply rotary embeddings
            q = apply_rotary_pos_emb(rotary_pos_emb, q)
            k = apply_rotary_pos_emb(rotary_pos_emb, k)
            if self.rotate_value:
                v = apply_rotary_pos_emb(rotary_pos_emb, v)

            attn_mask = (
                repeat(attn_mask, "b m -> b h n m", h=self.heads, n=q.size(1))
                if attn_mask is not None
                else None
            )
            attn_bias = (
                attn_mask.float().masked_fill(attn_mask, float("-inf"))
                if attn_mask is not None
                else None
            )

            # scaling is done by default
            out = xops.memory_efficient_attention(
                q, k, v, attn_bias=attn_bias, p=self.dropout
            )

            if self.rotate_value:
                out = apply_rotary_pos_emb(-rotary_pos_emb, out)

            # project back to output
            out = rearrange(out, "b n h d -> b n (h d)")
            out = self.to_out(out)
        else:
            q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
            k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
            v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)

            # apply rotary embeddings
            q = apply_rotary_pos_emb(rotary_pos_emb, q, dim=1)
            k = apply_rotary_pos_emb(rotary_pos_emb, k, dim=1)
            if self.rotate_value:
                v = apply_rotary_pos_emb(rotary_pos_emb, v, dim=1)

            # attention mask
            attn_mask = (
                rearrange(attn_mask, "b n -> b () () n")
                if attn_mask is not None
                else None
            )
            # perform attention, by default will use the optimal attention implementation
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=self.dropout
            )

            if self.rotate_value:
                out = apply_rotary_pos_emb(-rotary_pos_emb, out, dim=1)

            # project back to output
            out = rearrange(out, "b h n d -> b n (h d)")
            out = self.to_out(out)
        return out
