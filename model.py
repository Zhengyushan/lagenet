
"""
Modified from the code of
https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
"""

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, att_mask=None, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

               
class LAGEModule(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads > 0 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        if self.heads > 0:
            self.attend = nn.Softmax(dim = -1)
            self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.gcn_embed = nn.Linear(dim, dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim + dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, adj, att_mask=None):
        b, n, _, h = *x.shape, self.heads

        out = self.gcn_embed(x)
        out = einsum('b n j, b j d -> b n d', adj, out)

        if self.heads > 0:
            qkv = self.to_qkv(x).chunk(3, dim = -1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

            dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
            
            if att_mask is not None:
                attn = self.attend(dots.masked_fill(att_mask, torch.tensor(-1e9)))
            else:
                attn = self.attend(dots)

            att_out = einsum('b h i j, b h j d -> b h i d', attn, v)
            att_out = rearrange(att_out, 'b h n d -> b n (h d)')

            out = torch.cat((att_out, out), dim=-1)

        return self.to_out(out)


class LAGEBlock(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                LAGEModule(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)),
            ]))
        self.h = heads

    def forward(self, x, adj, mask=None):
        att_mask = einsum('b i d, b j d -> b i j', mask.float(), mask.float())
        att_mask = repeat(att_mask.unsqueeze(1), 'b () i j -> b h i j', h = self.h) < 0.5

        for pn, attn, ff in self.layers:
            x = pn(x)
            x = attn(x, adj, att_mask) + x
            x = ff(x) + x
        return x


class LAGENet(nn.Module):
    def __init__(self, num_patches, patch_dim, num_classes, num_hash_bits, dim, depth, heads, mlp_dim, pos_embed=True, pool = 'cls', dim_head = 64, dropout = 0.5, emb_dropout = 0.):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Linear(patch_dim, dim)

        self.trainable_pe = pos_embed
        if self.trainable_pe:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.lagcn = LAGEBlock(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.hash_norm = nn.LayerNorm(dim)
        self.hash_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_hash_bits),
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, node_features, adj, de=None, mask=None):
        x = self.to_patch_embedding(node_features)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)

        if self.trainable_pe:
            x += self.pos_embedding[:, :n]
        else:
            x += de
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = self.dropout(x)

        x = self.lagcn(x, adj, mask)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
 
        return self.mlp_head(x), torch.tanh(self.hash_head(x))

    def get_weights(self):
        return self.hash_head[1].weight