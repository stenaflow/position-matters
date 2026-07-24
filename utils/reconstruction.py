import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# SSIM
# ─────────────────────────────────────────────────────────────

def _gaussian_kernel(window_size: int = 11, sigma: float = 1.5,
                     device=None, dtype=None) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    k2d = torch.outer(g, g)
    k2d = k2d / k2d.sum()
    return k2d.view(1, 1, window_size, window_size)


def compute_ssim(img1_chw: torch.Tensor, img2_chw: torch.Tensor,
                 window_size: int = 11, sigma: float = 1.5,
                 data_range: float = 1.0, k1: float = 0.01,
                 k2: float = 0.03) -> float:
    if img1_chw.shape != img2_chw.shape:
        raise ValueError(f"Shape mismatch: {tuple(img1_chw.shape)} vs {tuple(img2_chw.shape)}")
    if img1_chw.ndim != 3:
        raise ValueError(f"Expected (C,H,W), got {tuple(img1_chw.shape)}")

    x = img1_chw.unsqueeze(0)
    y = img2_chw.unsqueeze(0)
    c = x.shape[1]

    kernel = _gaussian_kernel(window_size, sigma, device=x.device, dtype=x.dtype)
    kernel = kernel.repeat(c, 1, 1, 1)

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    pad = window_size // 2

    mu_x = F.conv2d(x, kernel, padding=pad, groups=c)
    mu_y = F.conv2d(y, kernel, padding=pad, groups=c)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sx2 = F.conv2d(x * x, kernel, padding=pad, groups=c) - mu_x2
    sy2 = F.conv2d(y * y, kernel, padding=pad, groups=c) - mu_y2
    sxy = F.conv2d(x * y, kernel, padding=pad, groups=c) - mu_xy

    num = (2 * mu_xy + c1) * (2 * sxy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sx2 + sy2 + c2)
    return float((num / (den + 1e-12)).mean().item())
