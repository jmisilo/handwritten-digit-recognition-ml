import argparse


def init_parser_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs to train the model for.",
    )

    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=None,
        help="Batch size.",
    )

    parser.add_argument(
        "-nw",
        "--num_workers",
        type=int,
        default=None,
        help="Number of workers for the data loader.",
    )

    parser.add_argument(
        "-lr",
        "--lr",
        type=float,
        default=None,
        help="Learning rate.",
    )

    parser.add_argument(
        "-d",
        "--dropout",
        type=float,
        default=None,
        help="Dropout probability.",
    )

    args = parser.parse_args()

    return args
