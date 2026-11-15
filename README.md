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
