import argparse
import tempfile
from math import floor

import torch as t
import torchvision
from torch.utils.data import random_split
from utils import SPLIT_SEED, save_images

import medmnist
from medmnist import INFO


def prepare_MNIST():
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


def prepare_MedMNIST(data_flag, size):
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info["python_class"])
    classnames = [v for _, v in info["label"].items()]

    for split in ["train", "val", "test"]:
        out_dir = f"data/{data_flag}_{size}/{split}"

        print("Downloading...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset = DataClass(root=tmp_dir, split=split, download=True, size=size, mmap_mode="r")

        print("Dumping images...")
        save_images(dataset, out_dir, classnames)


def prepare_CIFAR10():
    out_dir = f"data/cifar10"

    for split in ["train", "test"]:
        print("Downloading...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset = torchvision.datasets.CIFAR10(root=tmp_dir, train=split == "train", download=True)

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


def prepare_CIFAR100():
    out_dir = f"data/cifar100"

    for split in ["train", "test"]:
        print("Downloading...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset = torchvision.datasets.CIFAR100(root=tmp_dir, train=split == "train", download=True)

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


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset", type=str, required=True, help="Dataset to download")
    argparser.add_argument("--size", type=int, default=28, help="Image size to download")

    SUPPORTED_DATASETS = ["mnist", "cifar10", "cifar100"]
    SUPPORTED_MEDMNIST_DATASETS = ["bloodmnist", "dermamnist", "pathmnist"]

    args = argparser.parse_args()

    if args.dataset == "mnist":
        prepare_MNIST()
    elif args.dataset == "cifar10":
        prepare_CIFAR10()
    elif args.dataset == "cifar100":
        prepare_CIFAR100()

    if args.dataset in SUPPORTED_MEDMNIST_DATASETS:
        prepare_MedMNIST(args.dataset, args.size)

    if args.dataset not in SUPPORTED_DATASETS + SUPPORTED_MEDMNIST_DATASETS:
        raise ValueError(f"Dataset {args.dataset} not supported")


if __name__ == "__main__":
    main()
