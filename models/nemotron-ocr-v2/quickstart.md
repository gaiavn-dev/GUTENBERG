# Quickstart

## Prerequisites

- **Python 3.12** (the package requires `>=3.12,<3.13`)
- **CUDA toolkit** with `nvcc` on `PATH` (the package compiles a CUDA C++ extension at install time)
- **A CUDA GPU** (or set `TORCH_CUDA_ARCH_LIST` to cross-compile, e.g. `TORCH_CUDA_ARCH_LIST="8.0 9.0"`)

The CUDA toolkit version must share the same **major version** as the CUDA
bindings in your PyTorch install (e.g. toolkit 12.4 with `torch+cu128` is fine;
toolkit 12.4 with `torch+cu130` will fail).

On Slurm clusters, run the install on a GPU node or load the CUDA module first:

```bash
module load cuda12.4/toolkit/12.4.1   # example; adjust for your cluster
export CUDA_HOME=/usr/local/cuda       # or wherever the toolkit lives
```

## Installation

Install PyTorch **first** with bindings matching your CUDA toolkit, then install
this package with `--no-build-isolation` so it builds the C++ extension against
your existing PyTorch:

```bash
# 1. Install PyTorch (adjust the index URL for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2. Install nemotron-ocr
cd nemotron-ocr
pip install --no-build-isolation -v .
```

> **Why `--no-build-isolation`?** Without it, pip creates a temporary build
> environment and installs the latest PyTorch from PyPI. That PyTorch's CUDA
> version may not match your system's `nvcc`, causing the C++ extension build
> to fail with a CUDA version mismatch error.

Verify the installation (the C++ extension must load without errors):

```bash
python -c "from nemotron_ocr.inference.pipeline_v2 import NemotronOCRV2; print('OK')"
```

## Usage

`NemotronOCRV2` is the recommended entry point for OCR inference:

```python
from nemotron_ocr.inference.pipeline_v2 import NemotronOCRV2

ocr = NemotronOCRV2()
predictions = ocr("ocr-example-input-1.png")

for pred in predictions:
    print(f"  - Text: '{pred['text']}', Confidence: {pred['confidence']:.2f}")
```

The level of detection merging can be adjusted with `merge_level`:

```python
ocr(image_path, merge_level="word")       # individual words
ocr(image_path, merge_level="sentence")   # merged into sentences
ocr(image_path, merge_level="paragraph")  # merged into paragraphs (default)
```

### Inference modes

```python
# Detector only — bounding boxes, no text (fastest, lowest memory)
ocr_det = NemotronOCRV2(detector_only=True)

# Skip relational — per-word text, no reading-order grouping
ocr_fast = NemotronOCRV2(skip_relational=True)

# Profiling — per-phase CUDA-synced timing in logs
ocr_profile = NemotronOCRV2(verbose_post=True)
```

### Example script

```bash
python example.py ocr-example-input-1.png
python example.py ocr-example-input-1.png --merge-level word
python example.py ocr-example-input-1.png --detector-only
python example.py ocr-example-input-1.png --skip-relational
```
