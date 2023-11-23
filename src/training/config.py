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

    dropout: float = 0.15
    seed: int = 909
    batch_size: int = 32
    epochs: int = 150
    lr: float = 3e-4
    train_size: float = 0.84
    val_size: float = 0.11
    k: float = 0.25
    num_workers: int = 2
