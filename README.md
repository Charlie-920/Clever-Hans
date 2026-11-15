# Clever Hans

COMP3620 CW3 project on shortcut learning in Fashion-MNIST.

## Local development

In the repository root, use [uv](https://docs.astral.sh/uv/getting-started/installation/) to install the locked dependencies in `.venv`:

```sh
uv sync --locked
uv run --locked python scripts/check_environment.py
```

The local environment uses Python 3.11 and supports Apple silicon macOS, x86-64 Linux, and x86-64 Windows. The check imports PyTorch, Torchvision, and IPython Kernel and runs a small CPU tensor operation. For local notebooks, choose the repository's `.venv` as the Python environment in your editor.

## Course environment

On the university system, select the course-provided `condaenv/dl2026` kernel and restart it before running the notebooks from the top. The local environment is for development. Use the course kernel for submission checks.

## Fashion-MNIST minimum data handoff

From the repository root, run the following in a notebook cell or Python session. Set `download=True` for the first run with internet access. For later or offline runs, use `download=False` and point `root` at the directory containing `FashionMNIST/raw/`.

```python
from torch.utils.data import DataLoader

from fashion_data import load_minimum_data

clean_train, patched_train, clean_test = load_minimum_data("data", download=True)
print(len(clean_train), len(patched_train), len(clean_test))  # 60000 60000 10000

images, labels = next(iter(DataLoader(patched_train, batch_size=64)))
print(images.shape, images.dtype, labels.shape)  # [64, 1, 28, 28], float32, [64]
```

The two training datasets share the same images, indices and labels. Every image in `patched_train` receives a 3 x 3 patch with pixel value `1.0`, at a fixed position for its class. The datasets return `(image, label)` pairs, with image tensors of shape `(1, 28, 28)` and values in `[0, 1]`. For validation, use the same training indices in both conditions and reserve the clean test set for final evaluation.

The preliminary patch layout in `fashion_data.py` places the patch centres at least six pixels apart. We screened edge positions on the training split and counted images with any pixel above `0.1` in the patch area. With the selected class-to-position mapping, that area was occupied in 1.39% of training images overall and at most 6.73% within any one class.

The dataset comes from the [Fashion-MNIST authors](https://github.com/zalandoresearch/fashion-mnist). On first download, the loader uses their HTTPS mirror and verifies each archive against the checksums distributed with Torchvision.
