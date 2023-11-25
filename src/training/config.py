"""
    Project's main config.
"""

import os
from dataclasses import dataclass


@dataclass
class Config:
    """
    Project's main config.
    """

    seed: int = 909
    epochs: int = 150
    batch_size: int = 32
    train_size: float = 0.87
    lr: float = 3e-4
    dropout: float = 0.15
    k: float = 0.25
    num_workers: int = 2
    checkpoint_every_n_epochs: int = 10
    checkpoint_dir: str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "weights", "checkpoints")
    )

    data_dir: str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data")
    )
