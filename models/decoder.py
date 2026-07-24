import torch.nn as nn


class AttackerDecoder(nn.Module):
    def __init__(self, embed_dim=768, output_channels=3, patch_size=16, img_size=224):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size

        self.proj = nn.Linear(embed_dim, embed_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),

            nn.Conv2d(32, output_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        if x.shape[1] > self.grid_size ** 2:
            x = x[:, 1:, :]
        x = self.proj(x)
        B = x.shape[0]
        x = x.transpose(1, 2).reshape(B, self.embed_dim, self.grid_size, self.grid_size)
        return self.decoder(x)
