import os

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor

from evaluate import init_parser_args
from training import Config
from utils import device, is_cuda, logger

config = Config()


args = init_parser_args()

data = MNIST(
    download=True,
    root=config.data_dir,
    transform=Compose([ToTensor()]),
    train=False,
)

test_loader = DataLoader(
    data,
    batch_size=config.batch_size,
    pin_memory=is_cuda,
    num_workers=config.num_workers if is_cuda else 0,
)

if __name__ == "__main__":
    # Load the latest model
    model_path = os.path.join(
        config.weights_dir, args.file or os.listdir(config.weights_dir)[-1]
    )

    model = torch.jit.load(model_path)
    model = model.to(device)

    total, correct = 0, 0

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    logger.info(
        f"Accuracy of the network on the {total} test images: {(100 * correct / total):.2f}%"
    )
