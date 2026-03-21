"""
cbam.py
Implementación del módulo de atención CBAM (Convolutional Block Attention Module).
Combina atención de canal (CAM) y atención espacial (SAM) secuencialmente.
Referencia: Woo et al., 2018 - CBAM: Convolutional Block Attention Module.
Usado exclusivamente por EfficientNetV2S_CBAM.
"""

import torch
import torch.nn as nn


# ── Channel Attention Module (CAM) ────────────────────────
class ChannelAttention(nn.Module):
    """
    Recalibra la importancia de cada canal mediante AvgPool + MaxPool
    procesados por una MLP compartida (Ec. 14 del documento).
    r = ratio de reducción = 16
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # MLP compartida con ratio de reducción r=16
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.shared_mlp(self.avg_pool(x).view(b, c))
        max_out = self.shared_mlp(self.max_pool(x).view(b, c))
        mc = self.sigmoid(avg_out + max_out)
        mc = mc.view(b, c, 1, 1)
        return x * mc


# ── Spatial Attention Module (SAM) ────────────────────────
class SpatialAttention(nn.Module):
    """
    Genera mapa de atención espacial mediante Conv7x7
    sobre AvgPool y MaxPool a lo largo del eje de canales (Ec. 16 del documento).
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Pooling a lo largo del eje de canales
        avg_out = torch.mean(x, dim=1, keepdim=True)   # AvgPoolc(F')
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # MaxPoolc(F')
        # Concatenar y aplicar Conv7x7
        concat = torch.cat([avg_out, max_out], dim=1)
        ms = self.sigmoid(self.conv(concat))
        return x * ms  # F'' = Ms(F') ⊗ F'


# ── CBAM completo ──────────────────────────────────────────
class CBAM(nn.Module):
    """
    Módulo CBAM completo: CAM → SAM secuencial.
    Entrada: tensor F de forma (B, C, H, W)
    Salida:  tensor F'' refinado con atención de canal y espacial
    """
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)  # Atención de canal
        x = self.spatial_attention(x)  # Atención espacial
        return x


if __name__ == "__main__":
    # Test rápido
    import torch
    x = torch.randn(2, 256, 14, 14)  # Batch=2, C=256, H=14, W=14
    cbam = CBAM(in_channels=256, reduction_ratio=16, kernel_size=7)
    out = cbam(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    assert x.shape == out.shape, "Error: shapes no coinciden"
    print("cbam.py cargado correctamente ✓")