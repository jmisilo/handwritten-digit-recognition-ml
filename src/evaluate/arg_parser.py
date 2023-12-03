import argparse


def init_parser_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default=None,
        help="Model file name [in /weights/ directory].",
    )

    args = parser.parse_args()

    return args
