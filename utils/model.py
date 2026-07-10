"""
model.py — Arquitectura UNet de mapas HEALPix proyectados a 2D.

    UNetDenoiser: U-Net de 4 niveles adaptado al paper Wang et al. 2022,
                  pero reducido para imágenes 64×48 (nside=16).
                  Mejor para preservar estructuras de gran escala del mapa.
"""

__all__ = ["ConvBlock", "UpBlock", "UNetDenoiser"]

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# Bloques reutilizables
# ==============================================================================

class ConvBlock(nn.Module):
    """
    Bloque convolucional estándar:  Conv -> BN -> PReLU  (× 2).

    El stride del primer conv controla si hay downsampling (stride=2) o no (stride=1).

    Parameters
    ----------
    in_ch       : int
                  Canales de entrada.
    out_ch      : int
                  Canales de salida.
    stride      : int
                  Stride del primer conv. Default: 1 (sin downsampling).
    kernel_size : int
                  Tamaño del kernel. Default: 3.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(out_ch),
            nn.Conv2d(out_ch, out_ch, kernel_size, stride=1, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """
    Bloque de upsample para el decoder del U-Net:
        Upsample ×2 -> concat(skip) -> ConvBlock.

    Uso de interpolación bilineal en lugar de ConvTranspose2d para evitar
    el efecto checkerboard en mapas con bordes de caras HEALPix.

    Parameters
    ----------
    in_ch   : int
              Canales provenientes del nivel inferior (antes del cat).
    skip_ch : int
              Canales del skip connection correspondiente.
    out_ch  : int
              Canales de salida del bloque.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

# ==============================================================================
# U-Net adaptado (Wang et al. 2022, 4 niveles)
# ==============================================================================

class UNetDenoiser(nn.Module):
    """
    U-Net de 4 niveles para denoising de mapas HEALPix proyectados a 2D.

    Adaptado del paper Wang et al. (2022) para imágenes (1, 64, 48) — nside=16.
    Las resoluciones intermedias en el encoder son:
        64×48 -> 32×24 -> 16×12 -> 8×6 -> bottleneck 4×3

    Diferencias respecto al paper original:
        · 4 niveles en lugar de 8 (apropiado para 64×48).
        · Upsample bilineal en lugar de ConvTranspose2d.
        · PReLU en lugar de ReLU.
        · Downsampling configurable: MaxPool 2×2 (como Wang et al.) o
          strided convolution (Springenberg et al. 2015).

    Parameters
    ----------
    in_ch        : int
                   Canales de entrada. Default: 1. Para extracción de RM con N frecuencias,
                   usar in_ch = 2*N (Q y U apilados).
    base_ch      : int
                   Canales en el primer nivel del encoder. Se duplican en cada nivel.
                   Default: 32  ->  32 / 64 / 128 / 256 / bottleneck 512.
    out_ch       : int | None
                   Canales de salida. Si es None se iguala a `in_ch` (comportamiento
                   clásico de denoiser). Para la tarea (Q,U) -> RM usar out_ch=1.
                   Default: None.  
    use_residual : bool
                   Si True, la red predice el ruido residual y la salida es input - ruido.
                   Requiere in_ch == out_ch. Para tareas image-to-image con distinta
                   dimensionalidad (como (Q,U)->RM) debe ser False. Default: True.
    use_maxpool  : bool
                   Si True, el downsampling del encoder se realiza con MaxPool2d 2×2
                   antes de cada ConvBlock (esquema de Wang et al. 2022). MaxPool
                   preserva mejor los picos de activación a costa de descartar el 75%
                   de la información por ventana.
                   Si False, el downsampling se integra en la primera convolución de
                   cada ConvBlock mediante stride=2 (Springenberg et al. 2015). La
                   strided convolution aprende qué combinación lineal de píxeles
                   retener, pero puede suavizar picos extremos.
                   Default: False.
    """

    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 32,
        out_ch: int | None = None,
        use_residual: bool = True,
        use_maxpool: bool = False,
    ):
        super().__init__()
        # Si out_ch es None, la red es isomorfa en canales (uso original de denoiser).
        if out_ch is None:
            out_ch = in_ch
 
        if use_residual and out_ch != in_ch:
            raise ValueError(
                f"'use_residual=True' requiere in_ch == out_ch, pero se recibieron in_ch={in_ch} y out_ch={out_ch}."
            )

        self.use_maxpool = use_maxpool
        self.use_residual = use_residual
        self.out_ch = out_ch

        ch1  = base_ch        # 32
        ch2  = base_ch * 2    # 64
        ch3  = base_ch * 4    # 128
        ch4  = base_ch * 8    # 256

        ch_bottleneck = base_ch * 16  # 512
 
        # Encoder
        if use_maxpool:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.enc1 = ConvBlock(in_ch, ch1)              # 64×48
            self.enc2 = ConvBlock(ch1, ch2)                # 32×24  (pool antes)
            self.enc3 = ConvBlock(ch2, ch3)                # 16×12
            self.enc4 = ConvBlock(ch3, ch4)                #  8×6

            self.bottleneck = ConvBlock(ch4, ch_bottleneck) #  4×3
        else:
            self.enc1 = ConvBlock(in_ch, ch1)                       # 64×48
            self.enc2 = ConvBlock(ch1, ch2, stride=2)               # 32×24
            self.enc3 = ConvBlock(ch2, ch3, stride=2)               # 16×12
            self.enc4 = ConvBlock(ch3, ch4, stride=2)               #  8×6

            self.bottleneck = ConvBlock(ch4, ch_bottleneck, stride=2) #  4×3

        # Decoder: cada nivel recupera resolución via upsample + skip connection
        self.dec4 = UpBlock(ch_bottleneck, ch4,  ch4)       #  8×6
        self.dec3 = UpBlock(ch4,  ch3,  ch3)                # 16×12
        self.dec2 = UpBlock(ch3,  ch2,  ch2)                # 32×24
        self.dec1 = UpBlock(ch2,  ch1,    ch1)              # 64×48

        # Cabeza de salida: proyecta de ch1 a in_ch (igual a out_ch si use_residual=True)
        self.head = nn.Conv2d(ch1, out_ch, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        """ Inicialización Kaiming para Conv2d y constante para BatchNorm2d."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, in_channels, H, W)
 
        Returns
        -------
        torch.Tensor, shape (batch, in_channels, H, W)
            Mapa reconstruido (o señal residual si use_residual=False).
        """
        if x.ndim != 4 or x.shape[1] != self.enc1.block[0].in_channels:
            raise ValueError(f"Input debe tener shape (batch, {self.enc1.block[0].in_channels}, H, W), pero se recibió: {tuple(x.shape)}")    
        
        # Encoder: guardamos activaciones para los skip connections
        s1 = self.enc1(x)
        if self.use_maxpool:
            s2 = self.enc2(self.pool(s1))
            s3 = self.enc3(self.pool(s2))
            s4 = self.enc4(self.pool(s3))
            b  = self.bottleneck(self.pool(s4))
        else:
            s2 = self.enc2(s1)
            s3 = self.enc3(s2)
            s4 = self.enc4(s3)
            b  = self.bottleneck(s4)

        # Decoder (skip connections via cat)
        d4 = self.dec4(b,  s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        out = self.head(d1)
        return x - out if self.use_residual else out

    @property
    def n_params(self) -> int:
        """Número total de parámetros entrenables en la red."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)