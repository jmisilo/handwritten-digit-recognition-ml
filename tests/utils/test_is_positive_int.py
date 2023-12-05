import pytest

from src.utils import is_positive_int


def test_is_positive_int():
    assert is_positive_int(1) == 1, "is_positive_int throws error on positive int"

    with pytest.raises(ValueError) as e:
        is_positive_int(0)

    assert (
        str(e.value) == "value must be int, greater than 0"
    ), "is_positive_int does not throw error on 0"

    with pytest.raises(ValueError) as e:
        is_positive_int(-1)

    assert (
        str(e.value) == "value must be int, greater than 0"
    ), "is_positive_int does not throw error on negative int"

    with pytest.raises(ValueError) as e:
        is_positive_int(1.1)

    assert (
        str(e.value) == "value must be int, greater than 0"
    ), "is_positive_int does not throw error on float"

    with pytest.raises(ValueError) as e:
        is_positive_int("1")

    assert (
        str(e.value) == "value must be int, greater than 0"
    ), "is_positive_int does not throw error on string"

    with pytest.raises(ValueError) as e:
        is_positive_int(None)

    assert (
        str(e.value) == "value must be int, greater than 0"
    ), "is_positive_int does not throw error on None"
