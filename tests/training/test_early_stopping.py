import pytest

from src.training import EarlyStopper


def test_early_stopper():
    """
    Test EarlyStopper class.
    """
    early_stopper = EarlyStopper(patience=2, min_delta=0)

    assert early_stopper.patience == 2, "Patience should be 2."
    assert early_stopper.min_delta == 0, "Min delta should be 0."

    with pytest.raises(AttributeError) as excinfo:
        EarlyStopper(patience=2.0, min_delta=0)

    assert "Patience must be an integer." in str(
        excinfo.value
    ), "Patience must be an integer."

    with pytest.raises(AttributeError) as excinfo:
        EarlyStopper(patience=-2, min_delta=0)

    assert "Patience must be a non-negative integer." in str(
        excinfo.value
    ), "Patience must be a non-negative integer."

    with pytest.raises(AttributeError) as excinfo:
        EarlyStopper(patience=2, min_delta=-0.1)

    assert "Min delta must be a non-negative float." in str(
        excinfo.value
    ), "Min delta must be a non-negative float."

    early_stopper = EarlyStopper(patience=1, min_delta=0.01)
    early_stopper.min_validation_loss = 0.1

    assert early_stopper.early_stop(1e-5) == False, "Early stop should be False."

    assert early_stopper.early_stop(0.10) == True, "Early stop should be True."
