import argparse
import os

import torch
from PIL import Image
from torchvision import transforms

from training import Config
from utils import device, is_cuda

config = Config()

parser = argparse.ArgumentParser()

parser.add_argument(
    "-i",
    "--image_path",
    type=str,
    required=True,
    help="Path to the image to predict",
)

parser.add_argument(
    "-m",
    "--model_path",
    type=str,
    default=None,
    help="Path to the model to use for prediction",
)

args = parser.parse_args()


model_path = args.model_path or os.path.join(
    config.weights_dir, os.listdir(config.weights_dir)[-1]
)

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")

if not os.path.exists(args.image_path):
    raise FileNotFoundError(f"Image not found at {args.image_path}")


if __name__ == "__main__":
    model = torch.jit.load(model_path)
    model = model.to(device)

    image = Image.open(args.image_path).convert("L")

    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ]
    )

    image = transform(image).unsqueeze(0)

    model.eval()

    with torch.no_grad():
        output = model(image)

    _, predicted = torch.max(output.data, 1)

    print(f"Predicted: {predicted.item()}")
