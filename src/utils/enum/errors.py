from enum import Enum


class ErrorMessages(Enum):
    NO_VAL_DATA = "No validation data provided"
    NO_OPTIMIZER = "No optimizer provided"
    NO_CRITERION = "No criterion provided"
    INVALID_EPOCH = "Invalid epoch values. Epochs must be int, greater than 0"
    INVALID_START_EPOCH = "Invalid start epoch values. Epochs must be int, greater than 0, and less than epochs"
    INVALID_NUM_WORKERS = "Invalid num_workers. It must be an integer, >= 0."