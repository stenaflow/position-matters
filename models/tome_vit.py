import torch
import torch.nn as nn
import libs.tome as tome


def _tome_reset(vit, r_value: int):
    from libs.tome.utils import parse_r
    vit.r = r_value
    vit._tome_info["r"] = parse_r(len(vit.blocks), r_value)
    vit._tome_info["size"] = None
    vit._tome_info["source"] = None


class ClientVitToMe(nn.Module):
    def __init__(self, vit, split_block_number: int,
                 trace_source: bool = False, prop_attn: bool = False,
                 _skip_patch: bool = False, enable_pos_embed: bool = True):
        super().__init__()
        if not _skip_patch:
            tome.patch.timm(vit, trace_source=trace_source, prop_attn=prop_attn)

        self.vit              = vit
        self.split_block_number = split_block_number
        self.patch_embed      = vit.patch_embed
        self.cls_token        = vit.cls_token
        self.pos_embed        = vit.pos_embed
        self.pos_drop         = vit.pos_drop
        self.enable_pos_embed = enable_pos_embed
        self.blocks           = nn.ModuleList(vit.blocks[:split_block_number])
        self._tome_info       = vit._tome_info

    def forward(self, x: torch.Tensor, r_value: int):
        _tome_reset(self.vit, r_value)

        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)
        if self.enable_pos_embed:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        return x, self._tome_info["size"], self._tome_info["source"]
