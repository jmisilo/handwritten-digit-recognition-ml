import datetime
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, RandomRotation, ToTensor
from tqdm import tqdm

from model import Model
from training import Config, LRWarmup

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


def get_loaders(train_data, val_data, config):
    num_workers = config.num_workers if is_cuda else 0
    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=is_cuda,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config.batch_size,
        num_workers=num_workers,
        pin_memory=is_cuda,
    )
    return train_loader, val_loader, test_loader


train_loader = DataLoader(
    train_data,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=config.num_workers if is_cuda else 0,
    pin_memory=is_cuda,
)
test_loader = DataLoader(
    val_data,
    batch_size=config.batch_size,
    num_workers=config.num_workers if is_cuda else 0,
    pin_memory=is_cuda,
)

model = Model(p=config.dropout).to(device)

lr_warmup = LRWarmup(epochs=config.epochs, max_lr=config.lr, k=config.k)

optimizer = optim.Adam(model.parameters(), lr=config.lr)
criterion = nn.CrossEntropyLoss()

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_warmup)
scaler = torch.cuda.amp.GradScaler()

for epoch in range(config.epochs):
    loop = tqdm(enumerate(train_loader), total=len(train_loader), position=0)

    loop.set_description(f"Epoch: {epoch + 1} | Loss: ---")
    total_loss = 0
    total_correct = 0

    cur_lr = optimizer.param_groups[0]["lr"]
    for batch_idx, (data, target) in loop:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        model.train()
        with torch.cuda.amp.autocast():
            output = model(data)
            loss = criterion(output, target)

        total_loss += loss.item()

        pred = output.argmax(dim=1, keepdim=True)
        total_correct += pred.eq(target.view_as(pred)).sum().item()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.3)

        scaler.step(optimizer)
        scaler.update()

        loop.set_description(
            f"Epoch {epoch + 1} | Loss: {total_loss / (batch_idx + 1):.3f} | Accuracy: {total_correct / ((batch_idx + 1) * config.batch_size) * 100:.2f}% | LR: {cur_lr:.7f}"
        )
        loop.set_postfix(loss=loss.item())

    scheduler.step()

    model.eval()

    # validation loop

    loop = tqdm(enumerate(test_loader), total=len(test_loader), position=0)
    loop.set_description(f"Validation Epoch: {epoch + 1} | Loss: ---")

    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loop:
            data, target = data.to(device), target.to(device)

            output = model(data)

            val_loss += criterion(output, target).item()

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            loop.set_description(
                f"Validation Epoch {epoch + 1} | Loss: {val_loss / (batch_idx + 1):.3f} | Accuracy: {correct / ((batch_idx + 1) * config.batch_size) * 100:.2f}%"
            )
            loop.set_postfix(loss=loss.item())

    val_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)

    if not (epoch % config.checkpoint_every_n_epochs):
        torch.save(
            model.state_dict(),
            os.path.join(
                config.checkpoint_dir, f"model_{epoch}_{datetime.datetime.now()}.pth"
            ),
        )

torch.save(
    model.state_dict(),
    os.path.join(config.checkpoint_dir, f"model_{datetime.datetime.now()}.pth"),
)

# test loop
test_loss = 0
test_correct = 0

for data, target in test_loader:
    data, target = data.to(device), target.to(device)

    output = model(data)

    test_loss += criterion(output, target).item()

    pred = output.argmax(dim=1, keepdim=True)

    test_correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)
accuracy = 100.0 * test_correct / len(test_loader.dataset)
accuracy = 100.0 * test_correct / len(test_loader.dataset)
accuracy = 100.0 * test_correct / len(test_loader.dataset)
