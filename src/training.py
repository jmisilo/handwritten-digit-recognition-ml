import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, RandomRotation, ToTensor

from model import Model
from training import Config, LRWarmup, MNISTTrainer
from utils import Metrics

config = Config()


is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

data = MNIST(
    download=True,
    root=config.data_dir,
    transform=Compose([ToTensor(), RandomRotation(90)]),
)

test_data = MNIST(
    download=True, root=config.data_dir, train=False, transform=Compose([ToTensor()])
)

train_size = int(len(data) * config.train_size)
val_size = len(data) - train_size

train_data, val_data = random_split(data, [train_size, val_size])


if __name__ == "__main__":
    model = Model(p=config.dropout).to(device)

    lr_warmup = LRWarmup(epochs=config.epochs, max_lr=config.lr, k=config.k)

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_warmup)
    scaler = torch.cuda.amp.GradScaler()

    trainer = MNISTTrainer(
        model=model,
        train_data=train_data,
        val_data=val_data,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        clip_grad_norm=config.clip_grad_norm,
        checkpoint_every_n_epochs=config.checkpoint_every_n_epochs,
        checkpoint_dir=config.checkpoint_dir,
        weights_dir=config.weights_dir,
        epochs=config.epochs,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=is_cuda,
        device=device,
        metrics=[Metrics.ACCURACY],
    )

    print("Training...")
    for train_metrics, val_metrics in trainer.train():
        print(
            "Training metrics: ",
            train_metrics,
            "\n",
            "Validation metrics: ",
            val_metrics,
        )

    # # test loop
    # test_loss = 0
    # test_correct = 0

    # for data, target in test_loader:
    #     data, target = data.to(device), target.to(device)

    #     output = model(data)

    #     test_loss += criterion(output, target).item()

    #     pred = output.argmax(dim=1, keepdim=True)

    #     test_correct += pred.eq(target.view_as(pred)).sum().item()

    # test_loss /= len(test_loader.dataset)
    # accuracy = 100.0 * test_correct / len(test_loader.dataset)