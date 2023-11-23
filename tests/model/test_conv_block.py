import torch
import torch.nn as nn

from src.model.conv_block import ConvBlock


def test_conv_block() -> None:
    """Test ConvBlock class."""
    conv_block = ConvBlock(1, 1, kernel_size=3, stride=1, padding=1)
    assert isinstance(conv_block, nn.Module), "ConvBlock is not a Module subclass."
    assert isinstance(conv_block.conv, nn.Conv2d), "conv_block.conv is not a Conv2d."
    assert isinstance(
        conv_block.bn, nn.BatchNorm2d
    ), "conv_block.bn is not a BatchNorm2d."
    assert isinstance(conv_block.relu, nn.ReLU), "conv_block.relu is not a ReLU."

    x = torch.rand(1, 1, 30, 30)
    assert conv_block(x).shape == x.shape, "ConvBlock forward pass is incorrect."

    conv_block = ConvBlock(1, 1, kernel_size=3, stride=1, padding=1, relu=False)
    assert isinstance(conv_block, nn.Module), "ConvBlock is not a Module subclass."
    assert isinstance(conv_block.conv, nn.Conv2d), "conv_block.conv is not a Conv2d."
    assert isinstance(conv_block.bn, nn.Identity), "conv_block.bn is not an Identity."
    assert isinstance(
        conv_block.relu, nn.Identity
    ), "conv_block.relu is not an Identity."

    assert conv_block(x).shape == x.shape, "ConvBlock forward pass is incorrect."
