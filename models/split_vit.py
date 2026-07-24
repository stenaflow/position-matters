import torch
import torch.nn as nn


class ClientVit(nn.Module):
    def __init__(self, vit, split_block_number: int, enable_pos_embed=True):
        super().__init__()
        self.patch_embed = vit.patch_embed
        self.cls_token   = vit.cls_token
        self.pos_embed   = vit.pos_embed
        self.pos_drop    = vit.pos_drop
        self.enable_pos_embed = enable_pos_embed
        self.blocks = nn.ModuleList(vit.blocks[:split_block_number])

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)
        if self.enable_pos_embed:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class ServerVit(nn.Module):
    def __init__(self, vit, split_block_number: int):
        super().__init__()
        self.blocks = nn.ModuleList(vit.blocks[split_block_number:])
        self.norm   = vit.norm

    def forward(self, tokens):
        x = tokens
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)
