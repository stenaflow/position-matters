import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, dim=768, num_heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class TokenToImageDecoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_dim=768,
                 decoder_dim=768, depth=4, num_heads=8):
        super().__init__()
        assert img_size % patch_size == 0

        self.num_patches = (img_size // patch_size) ** 2  # 196

        self.input_proj = nn.Linear(in_dim, decoder_dim) if in_dim != decoder_dim else nn.Identity()
        self.pos_embed  = nn.Parameter(torch.zeros(1, self.num_patches, decoder_dim))

        self.blocks = nn.Sequential(*[
            TransformerBlock(dim=decoder_dim, num_heads=num_heads)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(decoder_dim)
        self.head = nn.Linear(decoder_dim, in_dim)

    def forward(self, tokens):
        x = self.input_proj(tokens)
        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x)
