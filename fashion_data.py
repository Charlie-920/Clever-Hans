"""Load the clean and patched Fashion-MNIST splits for the first handoff."""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torchvision.datasets.utils import download_and_extract_archive


PATCH_SIZE = 3

# Zero-based (row, column) of each patch's top-left pixel, in class ID order.
PATCH_POSITIONS = (
    (25, 25),  # T-shirt/top
    (18, 0),   # Trouser
    (0, 25),   # Pullover
    (6, 25),   # Dress
    (25, 0),   # Coat
    (6, 0),    # Sandal
    (0, 0),    # Shirt
    (0, 12),   # Sneaker
    (0, 6),    # Bag
    (12, 0),   # Ankle boot
)

# Use the dataset authors' HTTPS mirror and torchvision's archive checksums.
FASHION_MNIST_MIRROR = (
    "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
)


def add_white_patch(image: Tensor, top_left: tuple[int, int], size: int = PATCH_SIZE) -> Tensor:
    """Return a copy of a [0, 1] image with a square of pixels set to 1."""
    if image.ndim != 3 or image.shape[0] != 1 or not image.is_floating_point():
        raise ValueError("Expected a floating-point grayscale image shaped (1, H, W)")

    row, col = top_left
    height, width = image.shape[-2:]
    if size <= 0 or row < 0 or col < 0 or row + size > height or col + size > width:
        raise ValueError("The patch must fit inside the image")

    patched = image.clone()
    patched[:, row : row + size, col : col + size] = 1.0
    return patched


class PatchedFashionMNIST(Dataset[tuple[Tensor, int]]):
    """Add a class-matched patch while keeping each clean image's index and label."""

    def __init__(self, clean: FashionMNIST) -> None:
        self.clean = clean

    def __len__(self) -> int:
        return len(self.clean)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        image, label = self.clean[index]
        return add_white_patch(image, PATCH_POSITIONS[label]), label


def prepare_fashion_mnist(root: str | Path) -> None:
    """Download and verify missing Fashion-MNIST archives, then extract them."""
    raw = Path(root) / "FashionMNIST" / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    for filename, checksum in FashionMNIST.resources:
        extracted = Path(filename).stem
        destination = raw / extracted
        if destination.is_file():
            continue

        with TemporaryDirectory(dir=raw) as staging:
            download_and_extract_archive(
                FASHION_MNIST_MIRROR + filename,
                staging,
                filename=filename,
                md5=checksum,
            )
            (Path(staging) / extracted).replace(destination)


def load_minimum_data(
    root: str | Path = "data", *, download: bool = False
) -> tuple[FashionMNIST, PatchedFashionMNIST, FashionMNIST]:
    """Return the clean training set, 100% patched training set, and clean test set."""
    if download:
        prepare_fashion_mnist(root)

    to_tensor = transforms.Compose(
        [transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float32)]
    )
    clean_train = FashionMNIST(root, train=True, transform=to_tensor)
    clean_test = FashionMNIST(root, train=False, transform=to_tensor)
    return clean_train, PatchedFashionMNIST(clean_train), clean_test
