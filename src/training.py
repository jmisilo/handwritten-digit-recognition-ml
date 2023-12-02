import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, RandomRotation, ToTensor

from model import Model
from training import Config, LRWarmup, MNISTTrainer
from utils import logger
from utils.enum import Metrics

load_dotenv()

config = Config()

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

data = MNIST(
    download=True,
    root=config.data_dir,
    transform=Compose([ToTensor(), RandomRotation(90)]),
)


train_size = int(len(data) * config.train_size)
val_size = len(data) - train_size

train_data, val_data = random_split(data, [train_size, val_size])

if __name__ == "__main__":
    model = Model(p=config.dropout).to(device)

    lr_warmup = LRWarmup(epochs=config.epochs, max_lr=config.lr, k=config.k)

    optimizer = optim.Adam(model.parameters(), config.lr)
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
        num_workers=config.num_workers if is_cuda else 0,
        pin_memory=is_cuda,
        device=device,
        metrics=[Metrics.ACCURACY],
    )

    logger.info("Starting training...")
    for epoch, (train_metrics, val_metrics) in enumerate(trainer.train()):
        pass
    logger.info("Training finished.")
