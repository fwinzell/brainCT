import torch
import torch.nn as nn
from torchsummary import summary
from collections.abc import Sequence
import numpy as np

from utils import ConvLayer, SkipConnection
from unets import EncoderBlock, DecoderBlock, InputBlock3d

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class GUNet(nn.Module):
    """
    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        channels: sequence of channels. Top block first. The length of `channels` should be no less than 2.
        strides: sequence of convolution strides. The length of `stride` should equal to `len(channels) - 1`.
        kernel_size: convolution kernel size, the value(s) should be odd. If sequence,
            its length should equal to dimensions. Defaults to 3.
        up_kernel_size: upsampling convolution kernel size, the value(s) should be odd. If sequence,
            its length should equal to dimensions. Defaults to 3.
        act: activation type and arguments. Defaults to PReLU.
        norm: feature normalization type and arguments. Defaults to instance norm.
        dropout: dropout ratio. Defaults to no dropout.
        bias: whether to have a bias term in convolution blocks. Defaults to True.
        adn_ordering: a string representing the ordering of activation (A), normalization (N), and dropout (D).
            Defaults to "NDA".
        mode: which convolution to use, ``conv`` for regular convolution, ``separable`` for separable convolution.
    """

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            channels: Sequence[int],
            strides: Sequence[int],
            kernel_size: Sequence[int] | int = 3,
            up_kernel_size: Sequence[int] | int = 3,
            act: str = "PReLU",
            norm: str = "instance",
            dropout: float = 0.0,
            bias: bool = True,
            adn_ordering: str = "NDA",
            mode: str = "conv",
            use_3d_input: bool = False,
            out_channels_3d: int = None,
    ) -> None:
        super().__init__()

        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError("the length of `strides` should equal to `len(channels) - 1`.")
        if delta > 0:
            print(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")
        if isinstance(kernel_size, Sequence) and len(kernel_size) != spatial_dims:
            raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence) and len(up_kernel_size) != spatial_dims:
            raise ValueError("the length of `up_kernel_size` should equal to `dimensions`.")

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.adn_ordering = adn_ordering
        self.mode = mode
        self.bias = bias

        self.use_3d_input = use_3d_input
        if self.use_3d_input:
            assert out_channels_3d is not None, "out_channels_3d must be specified if use_3d_input is True"
        self.out_channels_3d = out_channels_3d

        self.model = self._create_block(in_channels, out_channels, self.channels, self.strides, True)
        self.initialize_parameters()

    def initialize_parameters(self,
                              method_weights=nn.init.kaiming_normal_,
                              method_bias=nn.init.zeros_,
                              kwargs_weights={},
                              kwargs_bias={}
                              ):
        print("Initializing parameters")
        for module in self.modules():
            self.weight_init(module, method_weights, **kwargs_weights)  # initialize weights
            self.bias_init(module, method_bias, **kwargs_bias)  # initialize bias

    @staticmethod
    def weight_init(module, method, **kwargs):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            method(module.weight, **kwargs)  # weights

    @staticmethod
    def bias_init(module, method, **kwargs):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            method(module.bias, **kwargs)  # bias

    def _create_block(self,
                      inc: int, outc: int, channels: Sequence[int], strides: Sequence[int], is_top: bool) -> nn.Module:
        """
        Builds the UNet structure from the bottom up by recursing down to the bottom block, then creating sequential
        blocks containing the downsample path, a skip connection around the previous block, and the upsample path.

        Args:
            inc: number of input channels.
            outc: number of output channels.
            channels: sequence of channels. Top block first.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        c = channels[0]
        s = strides[0]

        subblock: nn.Module

        if len(channels) > 2:
            subblock = self._create_block(c, c, channels[1:], strides[1:], False)  # continue recursion down
            upc = c * 2
        else:
            # the next layer is the bottom so stop recursion, create the bottom layer as the sublock for this layer
            subblock = self._get_bottom_layer(c, channels[1])
            upc = c + channels[1]

        if self.use_3d_input and is_top:
            down = self._get_input3d_layer(inc, c, s)
        else:
            down = self._get_enc_layer(inc, c, s, is_top)  # create layer in downsampling path
        up = self._get_dec_layer(upc, outc, s, is_top)  # create layer in upsampling path

        return self._get_connection_block(down, up, subblock)

    def _get_connection_block(self, down_path: nn.Module, up_path: nn.Module, subblock: nn.Module) -> nn.Module:
        """
        Returns the block object defining a layer of the UNet structure including the implementation of the skip
        between encoding (down) and decoding (up) sides of the network.

        Args:
            down_path: encoding half of the layer
            up_path: decoding half of the layer
            subblock: block defining the next layer in the network.
        Returns: block for this layer: `nn.Sequential(down_path, SkipConnection(subblock), up_path)`
        """
        return nn.Sequential(down_path, SkipConnection(subblock), up_path)

    def _get_enc_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Returns the encoding part of a layer of the network. This typically will downsample data at some point
        in its structure. Its output is used as input to the next layer down and is concatenated with output from the
        next layer to form the input for the decode part of the layer.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        mod: nn.Module

        mod = EncoderBlock(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            mode=self.mode,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )
        return mod

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        """
        Returns the bottom or bottleneck layer at the bottom of the network linking encode to decode halves.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
        """
        return self._get_enc_layer(in_channels, out_channels, 1, False)

    def _get_dec_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Returns the decoding part of a layer of the network. This typically will upsample data at some point
        in its structure. Its output is used as input to the next layer up.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        conv: nn.Module

        conv = DecoderBlock(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            last_conv_only=is_top,
            mode=self.mode,
            adn_ordering=self.adn_ordering,
            subunits=1
        )
        return conv

    def _get_input3d_layer(self, in_channels: int, out_channels: int, strides: int) -> nn.Module:
        mod: nn.Module

        mod = InputBlock3d(
            self.dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            out_channels_3d=self.out_channels_3d,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            mode=self.mode,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
            xdinput=3,
        )

        return mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x