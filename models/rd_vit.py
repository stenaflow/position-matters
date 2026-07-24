import torch
import torch.nn as nn


class ClientVitRD(nn.Module):
    def __init__(self, vit, split_block_number: int, enable_pos_embed: bool = True):
        super().__init__()
        self.patch_embed        = vit.patch_embed
        self.cls_token          = vit.cls_token
        self.pos_embed          = vit.pos_embed
        self.pos_drop           = vit.pos_drop
        self.enable_pos_embed   = enable_pos_embed
        self.blocks             = nn.ModuleList(vit.blocks[:split_block_number])
        self.split_block_number = split_block_number

    def forward(self, x: torch.Tensor, r: int) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)
        if self.enable_pos_embed:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
            if r > 0:
                cls_tok = x[:, :1, :]
                patches = x[:, 1:, :]
                N       = patches.shape[1]
                n_keep  = max(N - r, 1)

                kept_list = []
                for b in range(B):
                    perm = torch.randperm(N, device=x.device)
                    kept_list.append(patches[b, perm[:n_keep]])
                kept = torch.stack(kept_list, dim=0)
                x = torch.cat([cls_tok, kept], dim=1)

        return x
