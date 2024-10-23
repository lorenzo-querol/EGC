import argparse
import tempfile

import medmnist
from medmnist import INFO

from datasets.utils import save_images


def main(args):
    data_flag = args.data_flag
    size = args.image_size

    info = INFO[data_flag]
    DataClass = getattr(medmnist, info["python_class"])
    classnames = [v for _, v in info["label"].items()]

    for split in ["train", "val", "test"]:
        out_dir = f"data/{data_flag}_{size}/{split}"

        print("Downloading...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset = DataClass(root=tmp_dir, split=split, download=True, size=size, mmap_mode="r")

        print("Dumping images...")
        save_images(dataset, classnames, out_dir)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_flag", type=str, default="bloodmnist")
    argparser.add_argument("--image_size", type=int, default=28)
    args = argparser.parse_args()

    main(args)
