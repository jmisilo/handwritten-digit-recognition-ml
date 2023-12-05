import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor

from training import Config


def model_test(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    is_cuda: bool,
    device: torch.device,
    config: Config = Config(),
):
    """
    Test model on MNIST dataset.

    Args:
        model (torch.nn.Module): Model to test.
        criterion (torch.nn.Module): Loss function.
        is_cuda (bool): Whether CUDA is available.
        device (torch.device): Device to use.
        config (Config): Config object.

    Returns:
        Tuple[float, float]: Test loss and accuracy.
    """
    test_data = MNIST(
        download=True,
        root=config.data_dir,
        train=False,
        transform=Compose([ToTensor()]),
    )

    test_loader = DataLoader(
        test_data,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers if is_cuda else 0,
        pin_memory=is_cuda,
    )

    model.eval()
    test_loss = 0
    test_correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            test_loss += criterion(output, target).item()

            pred = output.argmax(dim=1, keepdim=True)

            test_correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * test_correct / len(test_loader.dataset)

    return test_loss, accuracy
