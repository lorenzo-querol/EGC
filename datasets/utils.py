import os
from tqdm.auto import tqdm

SPLIT_SEED = 1

CIFAR10_CLASSES = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

MNIST_CLASSES = (
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
)

DATASETS_CLASSES = {
    "mnist": MNIST_CLASSES,
    "cifar10": CIFAR10_CLASSES,
}


def save_images(dataset: str, out_dir: str, classnames: tuple | None = None) -> None:
    """
    Save images from the dataset to the output directory.

    Args:
        dataset (str): The dataset to save images from.
        classes (tuple): The classes of the dataset.
        out_dir (str): The output directory to save images to.
    Returns:
        None
    """
    os.makedirs(out_dir, exist_ok=True)

    if classnames is None:
        classnames = DATASETS_CLASSES.get(dataset.dataset_name, None)

    prog_bar = tqdm(range(len(dataset)))

    for i in prog_bar:
        image, label = dataset[i]
        label = label[0]
        filename = os.path.join(out_dir, f"{classnames[label]}_{i:05d}.png")
        image.save(filename)