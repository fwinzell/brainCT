import torch
import torch.nn as nn
from collections.abc import Sequence
import numpy as np

class ConvLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dimensions: int = 2,
                 strides: int = 1,
                 kernel_size: Sequence[int] | int = 3,
                 act: str = "relu",
                 mode: str = "conv",
                 norm: str = "instance",
                 bias: bool = True,
                 dropout: float = 0.0,
                 adn_ordering: str = "NDA",
                 conv_only: bool = False,
                 padding: int = 1,
                 output_padding: Sequence[int] | int = 0,
                 ):
        super(ConvLayer, self).__init__()
        self._dims = dimensions  # Other than 2 is not implemented
        self._out_channels = out_channels
        self._in_channels = in_channels
        self._ordering = adn_ordering
        self._conv_only = conv_only

        if act.lower() == "relu":
            self._act = nn.ReLU()
        elif act.lower() == "prelu":
            self._act = nn.PReLU()
        elif act.lower() == "leakyrelu":
            self._act = nn.LeakyReLU()
        else:
            assert False, f'Unknown activation: "{act}"'

        if norm.lower() == "batch":
            self._norm = nn.BatchNorm2d(self._out_channels)
        elif norm.lower() == "instance":
            self._norm = nn.InstanceNorm2d(self._out_channels)
        else:
            assert False, f'Unknown normalization: "{norm}"'

        if mode.lower() == "conv":
            self._conv = nn.Conv2d(self._in_channels, self._out_channels, kernel_size=kernel_size, stride=strides,
                                   padding=padding, bias=bias)
        elif mode.lower() == "separable":
            self._conv = SeparableConv2d(self._in_channels, self._out_channels, kernel_size=kernel_size, bias=bias)
        elif mode.lower() == "transposed":
            self._conv = nn.ConvTranspose2d(self._in_channels, self._out_channels, kernel_size=kernel_size,
                                            stride=strides, padding=padding, bias=bias, output_padding=output_padding)
            self._pooling = False
        else:
            assert False, f'Unknown convolution mode: "{mode}"'

        self._drop = nn.Dropout2d(dropout)

        self._op_dict = {"A": self._act, "N": self._norm, "D": self._drop}


    def forward(self, x):
        y = self._conv(x)
        if self._conv_only:
            return y
        else:
            for op in self._ordering:
                y = self._op_dict[op](y)
            return y


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   groups=in_channels, bias=bias, padding=1)
        self.pointwise = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class SkipConnection(nn.Module):
    """
    Combine the forward pass input with the result from the given submodule
    """
    def __init__(self, submodule, dim: int = 1) -> None:
        """
        Args:
            submodule: the module defines the trainable branch.
            dim: the dimension over which the tensors are concatenated.
        """
        super().__init__()
        self.submodule = submodule
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.submodule(x)
        return torch.cat([x, y], dim=self.dim)


class AttentionBlock(nn.Module):
    def __init__(self, spatial_dims: int, f_int: int, f_g: int, f_l: int, dropout=0.0):
        super().__init__()
        self.W_g = nn.Sequential(
            ConvLayer(
                dimensions=spatial_dims,
                in_channels=f_g,
                out_channels=f_int,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            # Batch norm
            nn.BatchNorm2d(f_int)
        )

        self.W_x = nn.Sequential(
            ConvLayer(
                dimensions=spatial_dims,
                in_channels=f_l,
                out_channels=f_int,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            nn.BatchNorm2d(f_int)
        )

        self.psi = nn.Sequential(
            ConvLayer(
                dimensions=spatial_dims,
                in_channels=f_int,
                out_channels=1,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            nn.BatchNorm2d(1)
        )

        self.relu = nn.ReLU()

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi: torch.Tensor = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class AttentionLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        submodule: nn.Module,
        out_channels: int = None,
        spatial_dims: int = 2,
        up_kernel_size=3,
        strides=2,
        dropout=0.0,
        do_upconv: bool = True,
    ):
        super().__init__()
        self.do_upc = do_upconv
        if self.do_upc:
            assert out_channels is not None, "Must specify out_channels if upconv is True"
            self.upconv = ConvLayer(
                in_channels=out_channels,
                out_channels=in_channels,
                strides=strides,
                kernel_size=up_kernel_size,
                act="relu",
                adn_ordering="NDA",
                norm="batch",
                mode="transposed"
            )
        else:
            self.upconv = nn.Identity()

        self.attention = AttentionBlock(
            spatial_dims=spatial_dims, f_g=in_channels, f_l=in_channels, f_int=in_channels // 2
        )

        self.merge = ConvLayer(in_channels=2 * in_channels, out_channels=in_channels, dropout=dropout)
        self.submodule = submodule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fromlower = self.upconv(self.submodule(x))
        att = self.attention(g=fromlower, x=x)
        att_m: torch.Tensor = self.merge(torch.cat((att, fromlower), dim=1))
        return att_m

