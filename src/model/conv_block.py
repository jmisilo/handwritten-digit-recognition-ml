import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Convolutional block consisting of a convolutional layer, batch normalization layer,
    and ReLU activation function.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Kernel size of the convolutional layer.
        stride (int): Stride of the convolutional layer.
        padding (int): Padding of the convolutional layer.
        bias (bool): Whether to include bias in the convolutional layer.
        relu (bool): Whether to include ReLU activation function.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Returns the output of the convolutional block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
        relu: bool = True,
    ) -> None:
        """Initialize ConvBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size of the convolutional layer.
            stride (int): Stride of the convolutional layer.
            padding (int): Padding of the convolutional layer.
            bias (bool): Whether to include bias in the convolutional layer.
            relu (bool): Whether to include ReLU activation function.

        Returns:
            None
        """
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

        self.bn = (
            nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.01, affine=True)
            if relu
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True) if relu else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the convolutional block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x
