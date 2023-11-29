import pytest
import torch

from src.model import Model


def test_net():
    x = torch.rand(1, 1, 28, 28)

    model = Model(p=0)
    assert model(x).shape == (1, 10), "Model forward pass is incorrect."
    assert model.dropout.p == 0, "Model dropout probability is incorrect."

    model = Model(p=0.5)
    assert model.dropout.p == 0.5, "Model dropout probability is incorrect."

    x = torch.rand(3, 1, 28, 28)
    assert model(x).shape == (3, 10), "Model forward pass is incorrect."

    with pytest.raises(AssertionError) as e:
        model = Model(p=1.1)

    assert (
        str(e.value) == "Dropout probability must be between 0 and 1."
    ), "Model dropout probability cannot be greater than 1."

    with pytest.raises(AssertionError) as e:
        model = Model(p=-0.1)

    assert (
        str(e.value) == "Dropout probability must be between 0 and 1."
    ), "Model dropout probability cannot be less than 0."
