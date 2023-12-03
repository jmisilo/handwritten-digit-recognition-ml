import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from src.training import Config, MNISTTrainer
from src.utils.enum import ErrorMessages


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)


model = TestModel()

trainer_constructor_params = {
    "model": model,
    "train_data": MNIST(
        download=True,
        root=Config.data_dir,
        transform=ToTensor(),
    ),
    "criterion": nn.CrossEntropyLoss(),
    "optimizer": optim.Adam(model.parameters(), lr=0.001),
}


def test_trainer():
    trainer = MNISTTrainer(
        **trainer_constructor_params,
        batch_size=10,
    )

    assert trainer.model == model, "Model is not assigned correctly."

    # mock model metrics data to test trainer metrics functionalities
    trainer.correct = 100
    trainer.compound_loss = 100

    assert trainer._accuracy(9) == 100, "Accuracy is not calculated correctly."

    assert trainer._loss(9) == 10, "Loss is not calculated correctly."

    assert (
        trainer._current_lr() == 0.001
    ), "Current learning rate is not calculated correctly."


def test_trainer_params_validator():
    # num_workers
    trainer = MNISTTrainer(
        **trainer_constructor_params,
        num_workers=0,
    )

    assert trainer.num_workers == 0, "Model has not been created with num_workers=0."

    trainer = MNISTTrainer(
        **trainer_constructor_params,
        num_workers=1,
    )

    assert (
        trainer.num_workers == 1
    ), "num_workers has not been created correctly, with num_workers as positive int."

    with pytest.raises(ValueError) as e:
        trainer = MNISTTrainer(
            **trainer_constructor_params,
            num_workers=-1,
        )

    assert (
        str(e.value) == ErrorMessages.INVALID_NUM_WORKERS.value
    ), "num_workers has not been validated correctly, with num_workers as negative int."

    with pytest.raises(ValueError) as e:
        trainer = MNISTTrainer(
            **trainer_constructor_params,
            num_workers=1.1,
        )

    assert (
        str(e.value) == ErrorMessages.INVALID_NUM_WORKERS.value
    ), "num_workers has not been validated correctly, with num_workers as float."

    # criterion
    trainer = MNISTTrainer(
        **trainer_constructor_params,
    )

    assert (
        trainer
    ), "criterion has not been created correctly, with criterion as nn.CrossEntropyLoss()."

    with pytest.raises(ValueError) as e:
        trainer = MNISTTrainer(
            **trainer_constructor_params | {"criterion": None},
        )

    assert (
        str(e.value) == ErrorMessages.NO_CRITERION.value
    ), "trainer has not been created correctly, with criterion."

    # optimizer
    trainer = MNISTTrainer(
        **trainer_constructor_params,
    )

    assert trainer, "trainer has not been created correctly, with optimizer."

    with pytest.raises(ValueError) as e:
        trainer = MNISTTrainer(
            **trainer_constructor_params | {"optimizer": None},
        )

    assert (
        str(e.value) == ErrorMessages.NO_OPTIMIZER.value
    ), "optimizer has not been validated correctly, with optimizer as None."
