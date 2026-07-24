from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.position_predictor import ContextualPositionPredictor
from models.mae import TokenToImageDecoder
from models.decoder import AttackerDecoder


class SARA(nn.Module):
    def __init__(self, pos_pred, mae, decoder, conf_threshold: float = 0.0):
        super().__init__()
        self.pos_pred       = pos_pred
        self.mae            = mae
        self.decoder        = decoder
        self.conf_threshold = conf_threshold

    def forward(self, smashed: torch.Tensor) -> torch.Tensor:
        B, T, D = smashed.shape
        device  = smashed.device

        positions            = self.pos_pred(smashed)
        probs                = F.softmax(positions, dim=-1)
        confidence, pred_pos = probs.max(dim=-1)
        pred_pos[:, 0]       = 0

        to_scatter = smashed.clone()
        if self.conf_threshold > 0.0:
            low_conf = confidence < self.conf_threshold
            low_conf[:, 0] = False
            to_scatter[low_conf] = 0.0

        sort_order = confidence.argsort(dim=1)
        pred_pos   = pred_pos.gather(1, sort_order)
        to_scatter = to_scatter[torch.arange(B, device=device).unsqueeze(1), sort_order]

        batch_idx   = torch.arange(B, device=device).unsqueeze(1).expand(B, T)
        full_tokens = torch.zeros(B, 197, D, device=device)
        full_tokens[batch_idx, pred_pos] = to_scatter

        patch_padded = full_tokens[:, 1:, :]
        missing_mask = (patch_padded.abs().sum(dim=-1) == 0)

        mae_out      = self.mae(patch_padded)
        patch_filled = patch_padded.clone()
        patch_filled[missing_mask] = mae_out[missing_mask]

        tokens = torch.cat([full_tokens[:, :1, :], patch_filled], dim=1)
        return self.decoder(tokens)
