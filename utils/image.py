import os
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from config import IMAGENET_MEAN, IMAGENET_STD


def unnormalize(img_chw: torch.Tensor,
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD) -> torch.Tensor:
    if img_chw.ndim != 3:
        raise ValueError(f"Expected (C,H,W), got {tuple(img_chw.shape)}")
    mean_t = torch.tensor(mean, device=img_chw.device)[:, None, None]
    std_t  = torch.tensor(std,  device=img_chw.device)[:, None, None]
    return (img_chw * std_t + mean_t).clamp(0.0, 1.0)
