from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from torchvision import datasets, transforms
import torch

from access_pytorch.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

def mnist_loaders(train_kwargs, test_kwargs):
    logger.info(f"Creating MNIST train and test loaders with {train_kwargs} and {test_kwargs}...")
    train_data, test_data = pull_mnist()

    train_loader = torch.utils.data.DataLoader(train_data, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, **test_kwargs)

    logger.success("Train and test loaders created.")
    return train_loader, test_loader

def cifar10_loaders(train_kwargs, test_kwargs):
    logger.info(f"Creating CIFAR10 train and test loaders with {train_kwargs} and {test_kwargs}...")
    train_data, test_data = pull_cifar10()
    
    train_loader = torch.utils.data.DataLoader(train_data, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, **test_kwargs)

    logger.success("Train and test loaders created.")
    return train_loader, test_loader

def pull_mnist():
    logger.info("Pulling MNIST dataset...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    logger.info(f"Applying {transform}")

    train_data = datasets.MNIST(PROCESSED_DATA_DIR, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(PROCESSED_DATA_DIR, train=False, transform=transform)

    logger.success("MNIST dataset processed, split, and transformed.")
    return train_data, test_data

def pull_cifar10():
    logger.info("Pulling CIFAR10 dataset...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    logger.info(f"Applying {transform}")

    train_data = datasets.CIFAR10(PROCESSED_DATA_DIR, train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(PROCESSED_DATA_DIR, train=False, transform=transform)

    logger.success("CIFAR10 dataset processed, split, and transformed.")
    return train_data, test_data

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
