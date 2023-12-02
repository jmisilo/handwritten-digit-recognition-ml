import datetime
import os
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from utils import is_positive_int, logger
from utils.enum import ErrorMessages, Metrics

from . import EarlyStopper


class MNISTTrainer:
    """
    Trainer for MNIST classification.
    """

    def __init__(
        self,
        model: nn.Module,
        train_data: torch.utils.data.Dataset,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        val_data: torch.utils.data.Dataset = None,
        scheduler: optim.lr_scheduler = None,
        scaler: torch.cuda.amp.GradScaler = None,
        clip_grad_norm: float | None = None,
        checkpoint_every_n_epochs: int | None = None,
        checkpoint_dir: str = None,
        load_checkpoint_file: str = None,
        weights_dir: str = None,
        epochs: int = 10,
        start_epoch: int = 0,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        use_early_stopper: bool = False,
        patience: int = 1,
        min_delta: float = 0,
        device: str = "cpu",
        metrics: List[Metrics] = [Metrics.ACCURACY],
    ) -> None:
        """Constructor for MNISTTrainer.

        Args:
            model (torch.nn.Module): Model to train.
            train_data (torch.utils.data.Dataset): Training data.
            criterion (torch.nn.Module): Loss function.
            optimizer (torch.optim.Optimizer): Optimizer.
            val_data (torch.utils.data.Dataset, optional): Validation data. Defaults to None.
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler. Defaults to None.
            scaler (torch.cuda.amp.GradScaler, optional): Gradient scaler. Defaults to None.
            clip_grad_norm (float | None, optional): Maximum gradient norm. Defaults to None.
            checkpoint_every_n_epochs (int | None, optional): Checkpoint every n epochs. Defaults to None.
            checkpoint_dir (str, optional): Checkpoint directory. Defaults to None.
            load_checkpoint_file (str, optional): Checkpoint file to load. Defaults to None.
            weights_dir (str, optional): Weights directory. Defaults to None.
            epochs (int, optional): Number of epochs. Defaults to 10.
            start_epoch (int, optional): Starting epoch. Defaults to 0.
            batch_size (int, optional): Batch size. Defaults to 32.
            num_workers (int, optional): Number of workers. Defaults to 0.
            pin_memory (bool, optional): Pin memory. Defaults to False.
            device (str, optional): Device. Defaults to "cpu".
            metrics (list, optional): Metrics to track. Defaults to [Metrics.ACCURACY].
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.device = device

        self.criterion = criterion

        self.optimizer = optimizer

        self.scheduler = scheduler
        self.scaler = scaler
        self.clip_grad_norm = clip_grad_norm

        self.load_checkpoint_file = load_checkpoint_file

        self.model = model.to(self.device)

        self.epochs = epochs
        self.start_epoch = start_epoch
        self.current_epoch = start_epoch

        self.metrics = metrics

        self.checkpoint_every_n_epochs = checkpoint_every_n_epochs

        self.checkpoint_dir = checkpoint_dir
        self.weights_dir = weights_dir

        self.correct = 0
        self.compound_loss = 0

        self.early_stopper = (
            EarlyStopper(patience=patience, min_delta=min_delta)
            if use_early_stopper and val_data is not None
            else None
        )

        self.__validate_params()
        self._init_training_loaders(train_data, val_data)

        self.__wandb_api_key = os.environ.get("WANDB_API_KEY")
        self.__wandb_project_name = os.environ.get(
            "WANDB_PROJECT_NAME", "handwritten-digit-recognition-ml"
        )

    def __validate_params(self) -> None:
        """
        Validate parameters.

        Raises:
            ValueError: If batch_size is not a positive integer.
            ValueError: If num_workers is not a positive integer.
            ValueError: If criterion is None.
            ValueError: If optimizer is None.
            ValueError: If epochs is not a positive integer.
            ValueError: If start_epoch is not a positive integer.
            ValueError: If start_epoch is greater than epochs.
            ValueError: If checkpoint_every_n_epochs is not a positive integer.

        Returns:
            None
        """

        is_positive_int(self.batch_size, name="batch_size")

        if self.num_workers < 0 or not isinstance(self.num_workers, int):
            raise ValueError(ErrorMessages.INVALID_NUM_WORKERS.value)

        if self.criterion is None:
            raise ValueError(ErrorMessages.NO_CRITERION.value)

        if self.optimizer is None:
            raise ValueError(ErrorMessages.NO_OPTIMIZER.value)

        self.__validate_epoch()
        self.__validate_metrics()

        if self.checkpoint_every_n_epochs is not None:
            is_positive_int(
                self.checkpoint_every_n_epochs,
                custom_message="checkpoint_every_n_epochs",
            )

    def __validate_metrics(self) -> None:
        """
        Validate metrics.

        Raises:
            ValueError: If metrics is not a list.
            ValueError: If metrics is not of type Metrics.

        Returns:
            None
        """
        if not isinstance(self.metrics, list):
            raise ValueError("Metrics must be a list")

        for metric in self.metrics:
            if not isinstance(metric, Metrics):
                raise ValueError("Metrics must be of type Metrics")

    def __validate_epoch(self) -> None:
        """
        Validate epochs.

        Raises:
            ValueError: If epochs is not a positive integer.
            ValueError: If start_epoch is not a positive integer.
            ValueError: If start_epoch is greater than epochs.

        Returns:
            None
        """
        is_positive_int(self.epochs, custom_message=ErrorMessages.INVALID_EPOCH.value)

        if (
            self.start_epoch < 0
            or not isinstance(self.start_epoch, int)
            or self.start_epoch >= self.epochs
        ):
            raise ValueError(ErrorMessages.INVALID_START_EPOCH.value)

    def _accuracy(self, batch_steps: int) -> float:
        """
        Calculate accuracy.

        Args:
            batch_steps (int): Number of batch steps performed in the epoch so far.

        Returns:
            float: Accuracy.
        """
        batch_steps += 1
        is_positive_int(batch_steps, name="batch_steps")

        return self.correct / (batch_steps * self.batch_size) * 100

    def _loss(self, batch_steps: int) -> float:
        """
        Calculate average loss in the epoch so far.

        Args:
            batch_steps (int): Number of batch steps performed in the epoch so far.

        Returns:
            float: Average loss.
        """
        batch_steps += 1
        is_positive_int(batch_steps, name="batch_steps")

        return self.compound_loss / batch_steps

    def _current_lr(self) -> float:
        """
        Get current learning rate.

        Returns:
            float: Current learning rate.
        """
        return self.optimizer.param_groups[0]["lr"]

    def _load_checkpoint(self) -> None:
        """
        Load checkpoint to resume training.

        Returns:
            None
        """
        if self.load_checkpoint_file is not None:
            checkpoint = torch.load(self.load_checkpoint_file)

            self.start_epoch = checkpoint["epoch"]
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            logger.info(f"Loaded checkpoint: {self.load_checkpoint_file}")

    def _init_training_loaders(
        self,
        train_data: torch.utils.data.Dataset,
        val_data: torch.utils.data.Dataset | None,
    ) -> None:
        """
        Initialize training and validation loaders.

        Args:
            train_data (torch.utils.data.Dataset): Training data.
            val_data (torch.utils.data.Dataset | None): Validation data.

        Returns:
            None
        """

        self.train_loader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        self.val_loader = (
            None
            if val_data is None
            else DataLoader(
                val_data,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
        )

    def _metrics(self, batch_steps: int = None):
        """
        Return metrics.

        Args:
            batch_steps (int, optional): Number of batch steps performed in the epoch so far. Defaults to None.

        Returns:
            dict: Metrics.
        """

        if batch_steps is None:
            batch_steps = len(self.train_loader)

        return {
            "loss": self._loss(batch_steps),
            **(
                {"lr": self._current_lr()}
                if self.scheduler is not None and self.model.training
                else {}
            ),
            **(
                {Metrics.ACCURACY.value: self._accuracy(batch_steps)}
                if Metrics.ACCURACY in self.metrics
                else {}
            ),
        }

    def _epoch_description(self, batch_idx: int) -> str:
        """
        Return epoch description.

        Args:
            batch_idx (int): Batch index.

        Returns:
            str: Epoch description.
        """
        core = f"Epoch {self.current_epoch + 1} | Loss: {self._loss(batch_idx + 1):.3f}"

        if Metrics.ACCURACY in self.metrics:
            core += f" | Accuracy: {self._accuracy(batch_idx):.2f}%"

        return core

    def _save_checkpoint(
        self,
        current_date: str,
        train_metrics: dict,
        val_metrics: dict,
        run: wandb.run = None,
    ) -> None:
        """
        Save checkpoint.

        Args:
            current_date (str): Current date.
            train_metrics (dict): Training metrics.
            val_metrics (dict): Validation metrics.
            run (wandb.Run): Wandb run.

        Returns:
            None
        """

        if self.checkpoint_every_n_epochs and not (
            (self.current_epoch + 1) % self.checkpoint_every_n_epochs
        ):
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"model_{self.current_epoch}_{current_date}.pth",
            )

            torch.save(
                {
                    "epoch": self.current_epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                },
                checkpoint_path,
            )

            if self.__wandb_api_key is not None:
                art = wandb.Artifact(
                    name=f"checkpoint_{self.__wandb_project_name}_{self.current_epoch}_{current_date}",
                    type="checkpoint",
                    description="Model checkpoint",
                )

                art.add_file(checkpoint_path)

                run.log_artifact(art)

    def _save_final_model(self, current_date: str, run: wandb.run = None) -> None:
        """
        Save weights of final model, in TorchScript format.

        Args:
            current_date (str): Current date.
            run (wandb.Run): Wandb run.

        Returns:
            None
        """

        self.model.eval()
        # save in TorchScript format
        model_scripted = torch.jit.trace(
            self.model, (torch.rand(1, 1, 28, 28).to(self.device))
        )
        model_path = os.path.join(self.weights_dir, f"model_scripted_{current_date}.pt")

        model_scripted.save(model_path)

        if self.__wandb_api_key is not None:
            art = wandb.Artifact(
                name=f"model_{self.__wandb_project_name}_{current_date}",
                type="model",
                description="Model weights",
            )

            art.add_file(model_path)

            run.log_artifact(art)

    def _wandb_log_metrics(self, train_metrics: dict, val_metrics: dict) -> None:
        """
        Log metrics to wandb.

        Args:
            train_metrics (dict): Training metrics.
            val_metrics (dict): Validation metrics.

        Returns:
            None
        """

        if self.__wandb_api_key is not None:
            logger.info("Logging to wandb")
            wandb.log(
                {
                    "train/loss": train_metrics["loss"],
                    **(
                        {"train/accuracy": train_metrics["accuracy"]}
                        if Metrics.ACCURACY in self.metrics
                        else {}
                    ),
                    **(
                        {"train/lr": train_metrics["lr"]}
                        if self.scheduler is not None
                        else {}
                    ),
                    **(
                        {"val/loss": val_metrics["loss"]}
                        if val_metrics is not None
                        else {}
                    ),
                    **(
                        {"val/accuracy": val_metrics["accuracy"]}
                        if val_metrics is not None and Metrics.ACCURACY in self.metrics
                        else {}
                    ),
                }
            )

    def __wandb_log_prediction_table(
        self, data: torch.Tensor, target: torch.Tensor, output: torch.Tensor
    ) -> None:
        if self.__wandb_api_key is not None and not self.current_epoch:
            table = wandb.Table(columns=["image", "correct", "prediction", "score"])
            for i in range(len(data)):
                table.add_data(
                    wandb.Image(data[i].to("cpu").numpy() * 255),
                    target[i].to("cpu").item(),
                    output[i].argmax(dim=0).to("cpu").item(),
                    output[i].softmax(dim=0).max(dim=0).values.to("cpu").item(),
                )

            wandb.log({"val/prediction_table": table}, commit=False)

    def __scaler_backprop(self, loss: torch.Tensor) -> None:
        """
        Backpropagate using the scaler.

        Args:
            loss (torch.Tensor): Loss.

        Returns:
            None
        """

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.clip_grad_norm
        )

        self.scaler.step(self.optimizer)
        self.scaler.update()

    def __backprop(self, loss: torch.Tensor) -> None:
        """
        Backpropagate.

        Args:
            loss (torch.Tensor): Loss.

        Returns:
            None
        """
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.clip_grad_norm
        )

        self.optimizer.step()

    def training_step(self, data: torch.Tensor, target: torch.Tensor) -> None:
        """
        Training step.

        Args:
            data (torch.Tensor): Input tensor.
            target (torch.Tensor): Target tensor.

        Returns:
            None
        """
        data, target = data.to(self.device), target.to(self.device)

        self.optimizer.zero_grad()

        self.model.train()
        with torch.cuda.amp.autocast():
            output = self.model(data)
            loss = self.criterion(output, target)

        self.compound_loss += loss.item()

        pred = output.argmax(dim=1, keepdim=True)
        self.correct += pred.eq(target.view_as(pred)).sum().item()

        if self.scaler is not None:
            self.__scaler_backprop(loss)
        else:
            self.__backprop(loss)

    def validation_step(self, data: torch.Tensor, target: torch.Tensor) -> None:
        """
        Validation step.

        Args:
            data (torch.Tensor): Input tensor.
            target (torch.Tensor): Target tensor.

        Returns:
            None
        """
        if self.val_loader is None:
            raise ValueError(ErrorMessages.NO_VAL_DATA.value)

        data, target = data.to(self.device), target.to(self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(data)

        self.__wandb_log_prediction_table(data, target, output)

        self.compound_loss += self.criterion(output, target).item()

        pred = output.argmax(dim=1, keepdim=True)
        self.correct += pred.eq(target.view_as(pred)).sum().item()

    def training_epoch(self):
        """
        Training epoch.

        Returns:
            dict: Metrics.
        """
        self.correct = 0
        self.compound_loss = 0

        loop = tqdm(
            enumerate(self.train_loader), total=len(self.train_loader), position=0
        )

        loop.set_description(f"Epoch: {self.current_epoch + 1}")

        self.model.train()
        for batch_idx, (data, target) in loop:
            self.training_step(data, target)

            loop.set_description(self._epoch_description(batch_idx))

            loop.set_postfix(LR=self._current_lr())

        if self.scheduler is not None:
            self.scheduler.step()

        return self._metrics()

    def validation_epoch(self):
        """
        Validation epoch.

        Returns:
            dict: Metrics.
        """
        if self.val_loader is None:
            raise ValueError(ErrorMessages.NO_VAL_DATA.value)

        self.correct = 0
        self.compound_loss = 0

        loop = tqdm(enumerate(self.val_loader), total=len(self.val_loader), position=0)

        loop.set_description(f"Validation Epoch: {self.current_epoch + 1} | Loss: ---")

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in loop:
                self.validation_step(data, target)

                loop.set_description(f"Validation {self._epoch_description(batch_idx)}")

        return self._metrics(batch_steps=batch_idx)

    def _init_wandb_run(self, config: dict) -> wandb.run:
        if self.__wandb_api_key is not None:
            run = wandb.init(
                project=self.__wandb_project_name,
                config=config.__dict__ if config is not None else {},
            )

            run.watch(self.model, log="all")

            return run

        return None

    def train(self, config=None):
        """
        Training loop.

        Yields:
            dict: Training metrics.
            dict: Validation metrics.
        """
        self._load_checkpoint()
        run = self._init_wandb_run(config)

        for epoch in range(self.start_epoch, self.epochs):
            self.current_epoch = epoch

            train_metrics = self.training_epoch()
            val_metrics = (
                self.validation_epoch() if self.val_loader is not None else None
            )

            current_date = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")

            self._wandb_log_metrics(train_metrics, val_metrics)

            self._save_checkpoint(current_date, train_metrics, val_metrics, run)

            if self.early_stopper is not None:
                if self.early_stopper.early_stop(val_metrics["loss"]):
                    logger.info("Early stopping!")
                    break

            yield train_metrics, val_metrics

        self._save_final_model(current_date, run)
        run.finish()
