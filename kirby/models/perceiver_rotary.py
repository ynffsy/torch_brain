import logging

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
try:
    import xformers.ops as xops
except ImportError:
    logging.warning("xformers not installed. Won't use memory-efficient attention.")
    xops = None


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
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, t_min=1e-4, t_max=4.0):
        super().__init__()
        # inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        inv_freq = torch.zeros(dim // 2)
        inv_freq[:dim // 4] = 2 * torch.pi / (t_min * ((t_max / t_min) ** (torch.arange(0, dim // 2, 2).float() / (dim // 2))))

        # t_min * ((t_max / t_min) ** (torch.arange(0, rotated_dims, 2).float() / rotated_dims))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, timestamps):
        freqs = torch.einsum('..., f -> ... f', timestamps, self.inv_freq)
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2)
        return freqs


def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)	
    x1, x2 = x.unbind(dim = -1)	
    x = torch.stack((-x2, x1), dim = -1)	
    return rearrange(x, '... d r -> ... (d r)')


def apply_rotary_pos_emb(freqs, x, dim=2):
    if dim==1:
        freqs = rearrange(freqs, 'n ... -> n () ...')
    elif dim==2:
        freqs = rearrange(freqs, 'n m ... -> n m () ...')
    x = (x * freqs.cos()) + (rotate_half(x) * freqs.sin())
    return x


class RotaryCrossAttention(nn.Module):
    def __init__(self, dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, rotate_value=False):
        super().__init__()

        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else dim
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dropout = dropout
        self.rotate_value = rotate_value

        # build networks
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x_query, x_context, rotary_pos_emb_query, rotary_pos_emb_context, attn_mask=None):
        # normalize
        x_query = self.norm(x_query)
        x_context = self.norm_context(x_context)

        # calculate query, key, value
        q = self.to_q(x_query)
        k, v = self.to_kv(x_context).chunk(2, dim=-1)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        # apply rotary embeddings
        q = apply_rotary_pos_emb(rotary_pos_emb_query, q, dim=1)
        k = apply_rotary_pos_emb(rotary_pos_emb_context, k, dim=1)
        if self.rotate_value:
            v = apply_rotary_pos_emb(rotary_pos_emb_context, v, dim=1)

        # attention mask
        attn_mask = rearrange(attn_mask, 'b n -> b () () n') if attn_mask is not None else None
        # perform attention, by default will use the optimal attention implementation
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout)

        if self.rotate_value:
            out = apply_rotary_pos_emb(-rotary_pos_emb_query, out, dim=1)

        # project back to output
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class RotarySelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, use_memory_efficient_attn=True, rotate_value=False):
        super().__init__()

        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
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
            q = rearrange(q, 'b n (h d) -> b n h d', h=self.heads)
            k = rearrange(k, 'b n (h d) -> b n h d', h=self.heads)
            v = rearrange(v, 'b n (h d) -> b n h d', h=self.heads)

            # apply rotary embeddings
            q = apply_rotary_pos_emb(rotary_pos_emb, q) 
            k = apply_rotary_pos_emb(rotary_pos_emb, k)
            if self.rotate_value:
                v = apply_rotary_pos_emb(rotary_pos_emb, v)

            attn_mask = repeat(attn_mask, 'b m -> b h n m', h=self.heads, n=q.size(1)) if attn_mask is not None else None
            attn_bias = attn_mask.float().masked_fill(attn_mask, float("-inf")) if attn_mask is not None else None
            
            # scaling is done by default
            out = xops.memory_efficient_attention(q, k, v, attn_bias=attn_bias, p=self.dropout)

            if self.rotate_value:
                out = apply_rotary_pos_emb(-rotary_pos_emb, out)

            # project back to output
            out = rearrange(out, 'b n h d -> b n (h d)')
            out = self.to_out(out)
        else:
            q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
            k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
            v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

            # apply rotary embeddings
            q = apply_rotary_pos_emb(rotary_pos_emb, q, dim=1)
            k = apply_rotary_pos_emb(rotary_pos_emb, k, dim=1)
            if self.rotate_value:
                v = apply_rotary_pos_emb(rotary_pos_emb, v, dim=1)
            

            # attention mask
            attn_mask = rearrange(attn_mask, 'b n -> b () () n') if attn_mask is not None else None
            # perform attention, by default will use the optimal attention implementation
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout)

            if self.rotate_value:
                out = apply_rotary_pos_emb(-rotary_pos_emb, out, dim=1)

            # project back to output
            out = rearrange(out, 'b h n d -> b n (h d)')
            out = self.to_out(out)
        return out


class Embedding(nn.Embedding): 
    def __init__(self, *args, init_scale=0.02, **kwargs,):
        self.init_scale = init_scale
        super().__init__(*args, **kwargs)
        
    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.weight, mean=0, std=self.init_scale)
        self._fill_padding_idx_with_zero()


class PerceiverNM(nn.Module):
    def __init__(
            self,
            *,
            dim=512,
            dim_head=64,
            num_latents=64,
            depth=2,
            output_dim=1,
            max_num_units=4096,  # For unit embeddings
            cross_heads=1,
            self_heads=8,
            ffn_dropout=0.2,
            lin_dropout=0.4,
            atn_dropout=0.0,
            num_tasks=1,
            emb_init_scale=0.02,
            use_memory_efficient_attn=True,
    ):
        super().__init__()

        use_memory_efficient_attn = use_memory_efficient_attn and xops is not None

        # Embeddings
        self.unit_emb = Embedding(max_num_units, dim, init_scale=emb_init_scale)
        self.spike_type_emb = Embedding(4, dim, init_scale=emb_init_scale)
        self.task_emb = Embedding(num_tasks, dim, init_scale=emb_init_scale)
        self.latent_emb = Embedding(num_latents, dim, init_scale=emb_init_scale)
        self.rotary_emb = RotaryEmbedding(dim_head)

        self.dropout = nn.Dropout(p=lin_dropout)

        # Encoding transformer (q-latent, kv-input spikes)
        self.enc_atn = RotaryCrossAttention(dim=dim, heads=cross_heads, dropout=atn_dropout, dim_head=dim_head, rotate_value=True)
        self.enc_ffn = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim=dim, dropout=ffn_dropout)
        )

        # Processing transfomers (qkv-latent)
        self.proc_layers = nn.ModuleList([])
        for i in range(depth):
            self.proc_layers.append(nn.ModuleList([
                RotarySelfAttention(dim=dim, heads=self_heads, dropout=atn_dropout, dim_head=dim_head, 
                                    use_memory_efficient_attn=use_memory_efficient_attn, rotate_value=True),
                nn.Sequential(nn.LayerNorm(dim), FeedForward(dim=dim, dropout=ffn_dropout))
            ]))

        # Decoding transformer (q-task query, kv-latent)
        self.dec_atn = RotaryCrossAttention(dim=dim, heads=cross_heads, dropout=atn_dropout, dim_head=dim_head, rotate_value=False)
        self.dec_ffn = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim=dim, dropout=ffn_dropout)
        )

        # Output projection (linear regression)
        self.decoder_out = nn.Linear(dim, output_dim)

    def forward(
            self,
            spike_unit=None,            # (B, N_in)
            spike_timestamps=None,      # (B, N_in)
            spike_type=None,            # (B, N_in)
            input_mask=None,            # (B, N_in)
            latent_id=None,             # (B, N_latent)
            latent_timestamps=None,     # (B, N_latent)
            output_timestamps=None,     # (B, N_out)
            task_id=None,               # (B)
            **kwargs
    ):
        # create embeddings
        x_input = self.unit_emb(spike_unit) + self.spike_type_emb(spike_type)
        latents = self.latent_emb(latent_id)
        x_output = self.task_emb(task_id)

        # compute timestamp embeddings
        spike_timestamp_emb = self.rotary_emb(spike_timestamps)
        latent_timestamp_emb = self.rotary_emb(latent_timestamps)
        query_timestamp_emb = self.rotary_emb(output_timestamps)

        # Encoder
        latents = latents + self.enc_atn(latents, x_input, latent_timestamp_emb, spike_timestamp_emb, input_mask)
        latents = latents + self.enc_ffn(latents)

        # Process
        for self_attn, self_ff in self.proc_layers:
            latents = latents + self.dropout(self_attn(latents, latent_timestamp_emb))
            latents = latents + self.dropout(self_ff(latents))

        # Decode
        x_output = x_output + self.dec_atn(x_output, latents, query_timestamp_emb, latent_timestamp_emb)
        x_output = x_output + self.dec_ffn(x_output)

        # Output projection
        output = self.decoder_out(x_output)

        return output
