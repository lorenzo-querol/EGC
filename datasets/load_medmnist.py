import argparse
import os
import tempfile

from tqdm.auto import tqdm

import medmnist
from medmnist import INFO


def main(args):
    data_flag = args.data_flag
    size = args.image_size

    info = INFO[data_flag]
    DataClass = getattr(medmnist, info["python_class"])
    CLASSES = [v for _, v in info["label"].items()]

    for split in ["train", "val", "test"]:
        out_dir = f"{data_flag}_{size}_{split}"
        if os.path.exists(out_dir):
            print(f"Skipping split {split} since {out_dir} already exists.")
            continue

        print("Downloading...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset = DataClass(root=tmp_dir, split=split, download=True, size=size, mmap_mode="r")

        print("Dumping images...")
        os.mkdir(out_dir)
        for i in tqdm(range(len(dataset))):
            image, label = dataset[i]
            label = label[0]
            filename = os.path.join(out_dir, f"{CLASSES[label]}_{i:05d}.png")
            image.save(filename)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_flag", type=str, default="bloodmnist")
    argparser.add_argument("--image_size", type=int, default=28)
    args = argparser.parse_args()

    main(args)
