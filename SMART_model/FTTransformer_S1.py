"""
This code is partially adapted from:

Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko.
"Revisiting Deep Learning Models for Tabular Data."
NeurIPS 2021.
Official implementation: https://github.com/yandex-research/rtdl-revisiting-models
PyTorch implementation: https://github.com/lucidrains/tab-transformer-pytorch
"""


import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat
from typing import Literal,Optional


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1) 
        return x * F.gelu(gates) 

def FeedForward(dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult * 2),
        GEGLU(), 
        nn.Dropout(dropout), 
        nn.Linear(dim * mult, dim) 
    )

_LINFORMER_KV_COMPRESSION_SHARING = Literal['headwise', 'key-value']

class Attention(nn.Module): 
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64, 
        dropout = 0.,
        # Linformer arguments
        n_tokens: Optional[int] = None,
        linformer_kv_compression_ratio: Optional[float] = None,
        linformer_kv_compression_sharing: Optional[
            _LINFORMER_KV_COMPRESSION_SHARING
        ] = None,
    ):
        super().__init__()
        inner_dim = dim_head * heads 
        self.heads = heads
        self.scale = dim_head ** -0.5 

        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False) 
        self.to_out = nn.Linear(inner_dim, dim, bias = False) 

        self.dropout = nn.Dropout(dropout)
        
        if linformer_kv_compression_sharing is not None:
            def make_linformer_kv_compression():
                return nn.Linear(
                    n_tokens,
                    max(int(n_tokens*linformer_kv_compression_ratio),1),
                    bias = False,
                )
            self.key_compression = make_linformer_kv_compression()
            self.value_compression = (
                make_linformer_kv_compression()
                if linformer_kv_compression_sharing == 'headwise'
                else None
            )
        else:
            self.key_compression = None
            self.value_compression = None

    def forward(self, x):
        h = self.heads 

        x = self.norm(x) 

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        
        if self.key_compression is not None:
            k = self.key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = (
                self.key_compression
                if self.value_compression is None
                else self.value_compression
            )(v.transpose(1, 2)).transpose(1, 2)
            
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v)) 
        q = q * self.scale 

        sim = einsum('b h i d, b h j d -> b h i j', q, k) 
        attn = sim.softmax(dim = -1) 
        dropped_attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', dropped_attn, v) 
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)

        return out, attn

class Transformer(nn.Module):
    def __init__(
        self,
        dim, 
        depth, 
        heads,
        dim_head,
        attn_dropout,
        ff_dropout, 
        n_tokens: Optional[int] = None,
        linformer_kv_compression_ratio: Optional[float] = None,
        linformer_kv_compression_sharing: Optional[
            _LINFORMER_KV_COMPRESSION_SHARING
        ] = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList([]) 

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout, n_tokens = n_tokens,linformer_kv_compression_ratio=linformer_kv_compression_ratio,linformer_kv_compression_sharing=linformer_kv_compression_sharing),
                FeedForward(dim, dropout = ff_dropout),
            ]))

    def forward(self, x, return_attn = False): 
        post_softmax_attns = [] 

        for attn, ff in self.layers: 
            attn_out, post_softmax_attn = attn(x)
            post_softmax_attns.append(post_softmax_attn)

            x = attn_out + x 
            x = ff(x) + x

        if not return_attn:
            return x

        return x, torch.stack(post_softmax_attns)


class NumericalEmbedder(nn.Module):
    def __init__(self, dim, num_numerical_types):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim)) 
        self.biases = nn.Parameter(torch.randn(num_numerical_types, dim)) 

    def forward(self, x):
        x = rearrange(x, 'b n -> b n 1')
        return x * self.weights + self.biases


class FTTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_continuous, 
        dim,
        depth,
        heads,
        dim_head = 16,
        dim_out = 14, 
        attn_dropout = 0.,
        ff_dropout = 0.,
        n_tokens: Optional[int] = None,
        linformer_kv_compression_ratio: Optional[float] = None,
        linformer_kv_compression_sharing: Optional[
            _LINFORMER_KV_COMPRESSION_SHARING
        ] = None,
    ):
        super().__init__()
       
        self.num_continuous = num_continuous

        if self.num_continuous > 0:
            self.numerical_embedder = NumericalEmbedder(dim, self.num_continuous)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.transformer = Transformer(
            dim = dim, 
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            n_tokens=n_tokens,
            linformer_kv_compression_ratio=linformer_kv_compression_ratio,
            linformer_kv_compression_sharing=linformer_kv_compression_sharing
        )

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim_out) 
        )

        self.multilabelclassifier = nn.Linear(dim, dim_out)

    def forward(self, x_numer, return_attn = True): 
        xs = []
        
        if self.num_continuous > 0:
            x_numer = self.numerical_embedder(x_numer)

            xs.append(x_numer)


        x = torch.cat(xs, dim = 1)

        b = x.shape[0]
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim = 1)


        x, attns = self.transformer(x, return_attn = True)

        x = x[:, 0]
        cls_token = x

        logits = self.multilabelclassifier(x)

        if not return_attn:
            return logits, cls_token

        return logits, cls_token, attns