class EarlyStopper:
    """
    Early stopping class to stop training if the validation loss does not improve for a certain number of epochs.

    Args:
        patience (int): Number of epochs to wait for the validation loss to improve.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.

    Attributes:
        patience (int): Number of epochs to wait for the validation loss to improve.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        counter (int): Number of epochs since the validation loss improved.
        min_validation_loss (float): Minimum validation loss observed so far.

    Methods:
        early_stop: Checks if the validation loss has not improved for a certain number of epochs.
    """

    def __init__(self, patience: int = 1, min_delta: float = 0):
        """
        Constructor method. Initializes the EarlyStopper class.

        Args:
            patience (int, optional): Number of epochs to wait for the validation loss to improve
            . Defaults to 1.
            min_delta (float, optional): Minimum change in the monitored quantity to qualify as an improvement. Defaults to 0.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

        self.__validate_params()

    def __validate_params(self):
        if not isinstance(self.patience, int):
            raise AttributeError("Patience must be an integer.")

        if not isinstance(self.min_delta, float) and not isinstance(
            self.min_delta, int
        ):
            raise AttributeError("Min delta must be a float or an integer.")

        if self.patience < 0:
            raise AttributeError("Patience must be a non-negative integer.")

        if self.min_delta < 0:
            raise AttributeError("Min delta must be a non-negative float.")

    def early_stop(self, validation_loss: float):
        """
        Checks if the validation loss has not improved for a certain number of epochs.

        Args:
            validation_loss (float): Validation loss.

        Returns:
            bool: True if the validation loss has not improved for a certain number of epochs, False otherwise.
        """

        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1

            if self.counter >= self.patience:
                return True
        return False
