"""Check that the local development environment can run PyTorch on the CPU."""

import platform

import ipykernel
import torch
import torchvision


def main() -> None:
    matrix = torch.eye(2, device="cpu")
    if not torch.equal(matrix @ matrix, matrix):
        raise RuntimeError("PyTorch CPU operation produced an unexpected result")

    print(f"Python {platform.python_version()}")
    print(f"PyTorch {torch.__version__}")
    print(f"Torchvision {torchvision.__version__}")
    print(f"IPython Kernel {ipykernel.__version__}")
    print("PyTorch CPU operation passed.")


if __name__ == "__main__":
    main()
