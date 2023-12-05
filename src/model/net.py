import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ConvBlock


class Model(nn.Module):
    """
    Model consisting of convolutional blocks, fully connected layers, and dropout layers, for MNIST classification.

    Args:
        p (float): Dropout probability.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Returns the output of the model.
    """

    def __init__(self, p=0):
        """
        Initialize Model.

        Args:
            p (int, optional): Dropout probability. Defaults to 0.

        Returns:
            None
        """
        super(Model, self).__init__()

        assert 0 <= p < 1, "Dropout probability must be between 0 and 1."

        self.c1 = ConvBlock(1, 16, kernel_size=4, stride=2)
        self.c2 = ConvBlock(16, 32, kernel_size=3, stride=1)
        self.c3 = ConvBlock(32, 1, kernel_size=2, stride=1)

        self.fc1 = nn.Linear(100, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_out = nn.Linear(256, 10)

        self.dropout = nn.Dropout(p)

        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights.

        Returns:
            None
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)

            if isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> nn.Module:
        """
        Return the output of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.dropout(self.c1(x))
        x = self.dropout(self.c2(x))
        x = self.dropout(self.c3(x))

        x = x.view(x.shape[0], -1)

        x = F.gelu(self.dropout(self.fc1(x)))
        x = F.gelu(self.dropout(self.fc2(x)))

        return self.fc_out(x)
