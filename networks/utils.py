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
                 act: str | tuple = "relu",
                 mode: str = "conv",
                 norm: str | tuple = "instance",
                 bias: bool = True,
                 dropout: float = 0.0,
                 adn_ordering: str = "NAD",
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

        act, act_params = self.split_args(act)
        norm, norm_params = self.split_args(norm)

        if act.lower() == "relu":
            self._act = nn.ReLU()
        elif act.lower() == "prelu":
            if act_params is not None:
                self._act = nn.PReLU(**act_params)
            else:
                self._act = nn.PReLU()
        elif act.lower() == "leakyrelu":
            if act_params is not None:
                self._act = nn.LeakyReLU(**act_params)
            else:
                self._act = nn.LeakyReLU()
        elif act.lower() == "sigmoid":
            self._act = nn.Sigmoid()
        else:
            assert False, f'Unknown activation: "{act}"'

        if norm.lower() == "batch":
            if norm_params is not None:
                self._norm = nn.BatchNorm2d(self._out_channels, **norm_params)
            else:
                self._norm = nn.BatchNorm2d(self._out_channels)
        elif norm.lower() == "instance":
            if norm_params is not None:
                self._norm = nn.InstanceNorm2d(self._out_channels, **norm_params)
            else:
                self._norm = nn.InstanceNorm2d(self._out_channels)
        elif norm.lower() == "none":
            self._norm = nn.Identity()
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

    @staticmethod
    def split_args(args):
        if isinstance(args, tuple):
            return args
        else:
            return args, None

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

class DoubleSkipConnection(nn.Module):
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

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        yy = self.submodule(x)
        if isinstance(yy, torch.Tensor):
            zz = [torch.cat([x, yy], dim=self.dim) for i in range(2)]
        else:
            zz = [torch.cat([x, y], dim=self.dim) for y in yy]
        return zz


class ParallelConnection(nn.Module):
    def __init__(self, down, conn, up1, up2) -> None:
        super().__init__()
        self.down = down
        self.conn = conn
        self.up1 = up1
        self.up2 = up2

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.down(x)
        x = self.conn(x)
        x_1 = self.up1(x[0])
        x_2 = self.up2(x[1])

        return [x_1, x_2]


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
        att_m: torch.Tensor = torch.cat((att, fromlower), dim=1) #self.merge(torch.cat((att, fromlower), dim=1))
        return att_m


class PlusDownBlock(nn.Module):
    def __init__(self,
                 dimensions: int,
                 in_channels: int,
                 out_channels: int,
                 strides: int = 1,
                 max_pool: bool = False,
                 kernel_size: Sequence[int] | int = 3,
                 act: str = "relu",
                 mode: str = "conv",
                 norm: str = "instance",
                 bias: bool = True,
                 dropout: float = 0.0,
                 adn_ordering: str = "NDA",
                 ):
        super().__init__()

        self._dims = dimensions
        if max_pool:
            self.strides = 1
            self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.strides = strides
            self.max_pool = nn.Identity()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Sequential()

        interch = in_channels
        interstr = self.strides
        for i in range(2):
            self.conv.add_module(f"conv{i}", ConvLayer(
                interch,
                self.out_channels,
                dimensions=self._dims,
                strides=interstr,
                kernel_size=kernel_size,
                act=act,
                mode=mode,
                norm=norm,
                bias=bias,
                dropout=dropout,
                adn_ordering=adn_ordering,
                conv_only=False,
            ))
            interch = self.out_channels
            interstr = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.max_pool(x)
        cx = self.conv(x)  # apply x to sequence of operations
        return cx


class PlusUpBlock(nn.Module):
    def __init__(self,
                 dimensions,
                 in_channels,
                 cat_channels,
                 out_channels,
                 act: str | tuple = "relu",
                 norm: str | tuple = "instance",
                 bias: bool = True,
                 dropout: float = 0.0,
                 adn_ordering: str = "NDA",
                 half_upconv: bool = True,
                 ):
        super().__init__()
        self._dims = dimensions  # Other than 2 is not implemented
        self.outc = out_channels
        self.catc = cat_channels
        self.inc = in_channels

        if half_upconv:
            self.upc = self.inc//2
        else:
            self.upc = self.inc
        # output_padding = strides + 2 * self.padding - np.array(kernel_size)

        self.upconv = ConvLayer(
            in_channels=self.inc,
            out_channels=self.upc,
            strides=2,
            kernel_size=2,
            padding=0,
            output_padding=0,
            conv_only=True,
            mode="transposed"
        )

        self.conv = nn.Sequential()

        cat = self.upc + self.catc
        for i in range(2):
            self.conv.add_module(
                f"conv{i}",
                ConvLayer(
                    in_channels=cat,
                    out_channels=self.outc,
                    strides=1,
                    kernel_size=3,
                    act=act,
                    adn_ordering=adn_ordering,
                    norm=norm,
                    bias=bias,
                    dropout=dropout,
                )
            )
            cat = self.outc

    def forward(self, x: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        x_0 = self.upconv(x)
        x = torch.cat((x_0, x_cat), dim=1)
        return self.conv(x)





