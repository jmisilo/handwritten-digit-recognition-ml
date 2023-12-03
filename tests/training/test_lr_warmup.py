from dataclasses import dataclass

from src.training import LRWarmup


@dataclass
class TestConfig:
    __test__ = False  # prevent pytest from collecting this class as a test

    epochs: int = 160
    lr: float = 3e-4
    k: float = 0.25


config = TestConfig()


def test_lr_warmup():
    lr_warmup = LRWarmup(epochs=config.epochs, max_lr=config.lr, k=config.k)

    assert lr_warmup.max_point == int(
        config.k * config.epochs
    ), "max_point is not correct"

    assert lr_warmup(0) == 0, "lr_warmup(0) is not correct"
    assert (
        lr_warmup(config.epochs * config.k) == 1
    ), "lr_warmup(max_point) is not correct"
    assert lr_warmup(config.epochs) == 0, "lr_warmup(epochs) is not correct"
