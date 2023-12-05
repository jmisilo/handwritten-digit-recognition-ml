"""
    Learning rate scheduler with warmup utility.
"""


class LRWarmup:
    """
    Self-made learning rate scheduler with warmup.
    """

    def __init__(self, epochs, max_lr, k):
        """Learning rate scheduler with warmup utility constructor.

        Args:
            epochs (int): Number of epochs.
            max_lr (float): Maximum learning rate.
            k (float): Percentage of epochs to warmup.

        Raises:
            AssertionError: If k is not between 0.05 and 0.95.

        Returns:
            None
        """
        assert k < 0.95 and k > 0.05, "k must be between 0.05 and 0.95"
        self.epochs = epochs
        self.max_lr = max_lr
        self.max_point = int(k * self.epochs)

    def __call__(self, epoch: int) -> float:
        """Learning rate scheduler.

        Args:
            epoch (int): Current epoch.

        Returns:
            float: Learning rate.
        """
        return self.lr_warmup(epoch)

    def lr_warmup(self, epoch: int) -> float:
        """Learning rate warmup function.

        Args:
            epoch (int): Current epoch.

        Returns:
            float: Learning rate.
        """
        a_1 = self.max_lr / self.max_point
        a_2 = self.max_lr / (self.max_point - self.epochs)

        b = -a_2 * self.epochs

        return min(a_1 * epoch, a_2 * epoch + b) / self.max_lr


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch
    import torch.optim as optim

    epochs = 60
    k = 0.25
    max_lr = 5.5e-4

    lr_warmup = LRWarmup(epochs, max_lr, k)

    optimizer = optim.Adam([torch.rand(12, 1)], max_lr)
    lr_warmup = LRWarmup(epochs=epochs, max_lr=optimizer.defaults["lr"], k=k)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_warmup)

    lrs = []
    for epoch in range(epochs):
        lrs.append(scheduler.get_last_lr()[0])

        optimizer.step()
        scheduler.step()

    plt.plot(lrs)
    plt.show()
