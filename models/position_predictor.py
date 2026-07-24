import warnings

import torch.nn as nn


class LinearPositionPredictor(nn.Module):
    def __init__(self, d: int = 768, n: int = 197):
        super().__init__()
        self.head = nn.Linear(d, n)

    def forward(self, x):
        return self.head(x)


class MLPPositionPredictor(nn.Module):
    def __init__(self, d: int = 768, n: int = 197, hidden: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, n),
        )

    def forward(self, x):
        return self.net(x)


class ContextualPositionPredictor(nn.Module):
    def __init__(self, d: int = 768, n: int = 197,
                 proj_dim: int = 384, n_heads: int = 6,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(d, proj_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=n_heads,
            dim_feedforward=proj_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True",
                category=UserWarning,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(proj_dim, n)

    def forward(self, x):
        x = self.proj(x)
        x = self.transformer(x)
        return self.head(x)


_ARCH_MAP = {
    "linear":      LinearPositionPredictor,
    "mlp":         MLPPositionPredictor,
    "transformer": ContextualPositionPredictor,
}


def build_tpp(arch: str = "transformer") -> nn.Module:
    if arch not in _ARCH_MAP:
        raise ValueError(f"Unknown TPP arch '{arch}'. Choose from: {list(_ARCH_MAP)}")
    return _ARCH_MAP[arch]()
