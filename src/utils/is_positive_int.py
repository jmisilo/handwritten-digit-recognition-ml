def is_positive_int(x, *, name="value", custom_message=None):
    if isinstance(x, int) and x > 0:
        return x

    error_message = custom_message or f"{name} must be int, greater than 0"

    raise ValueError(error_message)
