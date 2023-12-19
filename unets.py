import torch
import torch.nn as nn
from torchsummary import summary
from monai.networks.nets import AttentionUnet
from collections.abc import Sequence
import numpy as np

from utils import ConvLayer, SkipConnection, SeparableConv2d, AttentionBlock, AttentionLayer

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class TorchUnet(nn.Module):
    # Warning this is shit
    def __init__(self, config, freeze=False):
        super(TorchUnet, self).__init__()

        self.model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                                    in_channels=3, out_channels=1, init_features=32, pretrained=True)

        # Replace layers
        # Input layer
        # self.model.encoder1.enc1conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        # self.model.encoder1.enc1norm1 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.model.encoder1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
        )

        # Output layer, add 2 more channels
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=(2, 2), stride=(2, 2), bias=True)
        self.model.conv = nn.Conv2d(32, 3, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        return self.model(x)


class InputBlock3d(nn.Module):
    """
    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        strides: convolution stride. Defaults to 1.
        kernel_size: convolution kernel size. Defaults to 3.
        xdinput: number of inputs. Defaults to 3.
        cat_dim: dimension to concatenate the inputs. Defaults to 1.
        adn_ordering: a string representing the ordering of activation, normalization, and dropout.
            Defaults to "NDA".
        act: activation type and arguments. Defaults to PReLU.
        norm: feature normalization type and arguments. Defaults to instance norm.
        dropout: dropout ratio. Defaults to no dropout.
        dilation: dilation rate. Defaults to 1.
        bias: whether to have a bias term. Defaults to True.
        last_conv_only: for the last subunit, whether to use the convolutional layer only.
            Defaults to False.
        padding: controls the amount of implicit zero-paddings on both sides for padding number of points
            for each dimension. Defaults to None.
    """

    def __init__(self,
                 dimensions: int,
                 out_channels: int,
                 out_channels_3d: int = 16,
                 in_channels: int = 3,
                 strides: int = 1,
                 kernel_size: Sequence[int] | int = 3,
                 xdinput: int = 3,
                 cat_dim: int = 1,
                 act: str = "relu",
                 mode: str = "conv",
                 norm: str = "instance",
                 bias: bool = True,
                 dropout: float = 0.0,
                 adn_ordering: str = "NDA",
                 conv_only: bool = False,
                 ):
        super(InputBlock3d, self).__init__()

        self._dims = dimensions
        self.strides = strides
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_channels_3d = out_channels_3d
        self.cat_dim = cat_dim

        # Create a ModuleList to hold the parallel convolutional layers
        self.conv_layers = nn.ModuleList()

        # Add convolutional layers to the ModuleList
        for i in range(xdinput):
            self.conv_layers.append(ConvLayer(
                self.in_channels,
                self.out_channels_3d,
                dimensions=self._dims,
                strides=self.strides,
                kernel_size=kernel_size,
                act=act,
                mode=mode,
                norm=norm,
                bias=bias,
                dropout=dropout,
                adn_ordering=adn_ordering,
                conv_only=conv_only,
            ))

        if self.out_channels != self.out_channels_3d * xdinput:
            self.residual = nn.Conv2d(self.out_channels_3d * xdinput, self.out_channels, kernel_size=1, stride=1,
                                      padding=0)
        else:
            self.residual = nn.Identity(self.out_channels)

    def forward(self, x):
        # Apply each convolutional layer in parallel
        parallel_outputs = [conv(torch.squeeze(x[:, i])) for i, conv in enumerate(self.conv_layers)]

        # Add a batch dimension if necessary
        if x.shape[0] == 1:
            parallel_outputs = [torch.unsqueeze(output, 0) for output in parallel_outputs]

        # Combine the outputs (you may use other methods to combine the outputs based on your task)
        combined_output = torch.cat(parallel_outputs, dim=self.cat_dim)
        res = self.residual(combined_output)

        return res


class EncoderBlock(nn.Module):
    """
    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        strides: convolution stride. Defaults to 1.
        kernel_size: convolution kernel size. Defaults to 3.
        subunits: number of convolutions. Defaults to 2.
        adn_ordering: a string representing the ordering of activation, normalization, and dropout.
            Defaults to "NDA".
        act: activation type and arguments. Defaults to PReLU.
        norm: feature normalization type and arguments. Defaults to instance norm.
        dropout: dropout ratio. Defaults to no dropout.
        dilation: dilation rate. Defaults to 1.
        bias: whether to have a bias term. Defaults to True.
        last_conv_only: for the last subunit, whether to use the convolutional layer only.
            Defaults to False.
        padding: controls the amount of implicit zero-paddings on both sides for padding number of points
            for each dimension. Defaults to None.
    """

    def __init__(self,
                 dimensions: int,
                 in_channels: int,
                 out_channels: int,
                 strides: int = 1,
                 kernel_size: Sequence[int] | int = 3,
                 subunits: int = 2,
                 act: str = "relu",
                 mode: str = "conv",
                 norm: str = "instance",
                 bias: bool = True,
                 dropout: float = 0.0,
                 adn_ordering: str = "NDA",
                 conv_only: bool = False,
                 ):
        super(EncoderBlock, self).__init__()

        self._dims = dimensions
        self.strides = strides
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Sequential()

        interch = in_channels
        interstr = self.strides
        for i in range(subunits):
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
                conv_only=conv_only,
            ))
            interch = self.out_channels
            interstr = 1

        # apply convolution to input to change number of output channels and size to match that coming from self.conv
        if np.prod(self.strides) != 1 or self.in_channels != self.out_channels:
            rkernel_size = kernel_size
            rpadding = 1

            if np.prod(self.strides) == 1:  # if only adapting number of channels a 1x1 kernel is used with no padding
                rkernel_size = 1
                rpadding = 0

            self.residual = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=rkernel_size, stride=strides,
                                      padding=rpadding)
        else:
            self.residual = nn.Identity(self.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)  # create the additive residual from x
        cx = self.conv(x)  # apply x to sequence of operations
        return cx + res  # add the residual to the output


class DecoderBlock(nn.Module):
    """
    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        strides: convolution stride. Defaults to 2.
        kernel_size: convolution kernel size. Defaults to 2.
        subunits: number of convolutions. Defaults to 2.
        adn_ordering: a string representing the ordering of activation, normalization, and dropout.
            Defaults to "NDA".
        act: activation type and arguments. Defaults to PReLU.
        norm: feature normalization type and arguments. Defaults to instance norm.
        dropout: dropout ratio. Defaults to no dropout.
        dilation: dilation rate. Defaults to 1.
        bias: whether to have a bias term. Defaults to True.
        last_conv_only: for the last subunit, whether to use the convolutional layer only.
            Defaults to False.
        padding: controls the amount of implicit zero-paddings on both sides for padding number of points
            for each dimension. Defaults to None.
    """

    def __init__(self,
                 dimensions: int,
                 in_channels: int,
                 out_channels: int,
                 strides: int = 2,
                 kernel_size: Sequence[int] | int = 2,
                 subunits: int = 2,
                 act: str = "relu",
                 mode: str = "conv",
                 norm: str = "instance",
                 bias: bool = True,
                 dropout: float = 0.0,
                 adn_ordering: str = "NDA",
                 last_conv_only: bool = False,
                 ):
        super(DecoderBlock, self).__init__()

        self._dims = dimensions
        self.strides = strides
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Sequential()
        self.padding = 1
        self.output_padding = strides + 2 * self.padding - np.array(kernel_size)
        self.upconv = ConvLayer(
            self.in_channels,
            self.out_channels,
            dimensions=self._dims,
            strides=self.strides,
            kernel_size=kernel_size,
            act=act,
            mode="transposed",
            norm=norm,
            bias=bias,
            dropout=dropout,
            adn_ordering=adn_ordering,
            conv_only=False,
            padding=self.padding,
            output_padding=self.output_padding,
        )

        # self.conv.add_module(f"upconv", self.upconv)

        for i in range(subunits):
            conv_only = last_conv_only and i == subunits - 1
            self.conv.add_module(f"conv{i}", ConvLayer(
                self.out_channels,
                self.out_channels,
                dimensions=self._dims,
                strides=1,
                kernel_size=kernel_size,
                act=act,
                mode=mode,
                norm=norm,
                bias=bias,
                dropout=dropout,
                adn_ordering=adn_ordering,
                conv_only=conv_only,
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        upx = self.upconv(x)
        cx = self.conv(upx)
        return cx


class UNet(nn.Module):
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


# Attention gated output unet 3.5d
class UNet3d_AG(UNet):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            out_channels_3d: int,
            channels: Sequence[int],
            strides: Sequence[int],
            n_inputs: int = 3,
            kernel_size: Sequence[int] | int = 3,
            up_kernel_size: Sequence[int] | int = 3,
            act: str = "PReLU",
            norm: str = "instance",
            dropout: float = 0.0,
            bias: bool = True,
            adn_ordering: str = "NDA",
            mode: str = "conv",
    ) -> None:
        super().__init__(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            out_channels_3d=out_channels_3d,
            channels=channels,
            strides=strides,
            kernel_size=kernel_size,
            up_kernel_size=up_kernel_size,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=bias,
            adn_ordering=adn_ordering,
            mode=mode,
            use_3d_input=True,
        )
        self.n_inputs = n_inputs
        self.out_channels_3d = out_channels_3d

        self.model = self._create_block(in_channels, out_channels, self.channels, self.strides, True)

    def _create_block(self,
                      inc: int, outc: int, channels: Sequence[int], strides: Sequence[int], is_top: bool) -> nn.Module:

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

        if is_top:
            down = self._get_input3d_layer(inc, c, s)
            up = self._get_dec_layer(upc, outc, s, is_top)
            attn = self._get_attention_layer(c, subblock)
            return nn.Sequential(down, attn, up)
        else:
            down = self._get_enc_layer(inc, c, s, is_top)  # create layer in downsampling path
            up = self._get_dec_layer(upc, outc, s, is_top)  # create layer in upsampling path
            return self._get_connection_block(down, up, subblock)

    def _get_attention_layer(self, in_channels: int, subblock: nn.Module) -> nn.Module:
        """
        Returns the attention layer of a layer of the network.

        Args:
            in_channels: number of input channels.
            subblock: Network
        """
        mod: nn.Module

        mod = AttentionLayer(
            in_channels,
            submodule=subblock,
            dropout=self.dropout,
            do_upconv=False
        )
        return mod


class UNet_DeepFusion(UNet):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            out_channels_3d: int,
            channels: Sequence[int],
            strides: Sequence[int],
            n_inputs: int = 3,
            kernel_size: Sequence[int] | int = 3,
            up_kernel_size: Sequence[int] | int = 3,
            act: str = "PReLU",
            norm: str = "instance",
            dropout: float = 0.0,
            bias: bool = True,
            adn_ordering: str = "NDA",
            mode: str = "conv",
    ) -> None:
        super().__init__(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            out_channels_3d=out_channels_3d,
            channels=channels,
            strides=strides,
            kernel_size=kernel_size,
            up_kernel_size=up_kernel_size,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=bias,
            adn_ordering=adn_ordering,
            mode=mode,
            use_3d_input=True,
        )




if __name__ == "__main__":
    unet = UNet(spatial_dims=2,
                in_channels=3,
                out_channels=3,
                channels=(24, 48, 96, 192, 384),
                strides=(2, 2, 2, 2),
                kernel_size=3,
                up_kernel_size=3,
                use_3d_input=True,
                out_channels_3d=8)

    unet_att = UNet3d_AG(in_channels=3,
                         out_channels=3,
                         out_channels_3d=8,
                         channels=(24, 48, 96, 192, 384),
                         strides=(2, 2, 2, 2),
                         kernel_size=3,
                         up_kernel_size=3)

    summary(unet.to(device), (3, 3, 256, 256))

    # torchunet = TorchUnet(config=None)

    # summary(torchunet.to(device), (3, 256, 256))
