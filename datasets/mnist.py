import tempfile
from math import floor

import torch as t
import torchvision
from torch.utils.data import random_split
from utils import SPLIT_SEED, save_images


def main():
    out_dir = f"data/mnist"

    for split in ["train", "test"]:
        print("Downloading...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset = torchvision.datasets.MNIST(root=tmp_dir, train=split == "train", download=True)

        if split == "train":
            train_size = floor(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=t.Generator().manual_seed(SPLIT_SEED))

            print("Dumping training images...")
            save_images(train_dataset, f"{out_dir}/train")

            print("Dumping validation images...")
            save_images(val_dataset, f"{out_dir}/val")
        else:
            print("Dumping test images...")
            save_images(dataset, f"{out_dir}/test")


if __name__ == "__main__":
    main()
