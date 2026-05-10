---
license: other
license_name: nvidia-open-model-license
license_link: >-
  https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
language:
- en
- zh
- ja
- ko
- ru
pipeline_tag: image-to-text
arxiv: None
tags:
- image
- ocr
- object recognition
- text recognition
- layout analysis
- ingestion
- multilingual
---

# Nemotron OCR v2

## **Model Overview**

### **Description**

Nemotron OCR v2 is a state-of-the-art multilingual text recognition model designed for robust end-to-end optical character recognition (OCR) on complex real-world images. It integrates three core neural network modules: a detector for text region localization, a recognizer for transcription of detected regions, and a relational model for layout and structure analysis.

This model is optimized for a wide variety of OCR tasks, including multi-line, multi-block, and natural scene text, and it supports advanced reading order analysis via its relational model component. Nemotron OCR v2 supports multiple languages and has been developed to be production-ready and commercially usable, with a focus on speed and accuracy on both document and natural scene images.

Nemotron OCR v2 is part of the NVIDIA NeMo Retriever collection, which provides state-of-the-art, commercially-ready models and microservices optimized for the lowest latency and highest throughput. It features a production-ready information retrieval pipeline with enterprise support. The models that form the core of this solution have been trained using responsibly selected, auditable data sources. With multiple pre-trained models available as starting points, developers can readily customize them for domain-specific use cases, such as information technology, human resource help assistants, and research and development assistants.

This model is ready for commercial use.

### **License/Terms of use**

The use of this model is governed by the [NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) and the use of the post-processing scripts are licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt).

### Release Date:  <br>
Hugging Face (this repo): [nvidia/nemotron-ocr-v2](https://huggingface.co/nvidia/nemotron-ocr-v2) <br>
Build.Nvidia.com 04/15/2026 via [https://build.nvidia.com/nvidia/nemotron-ocr-v2](https://build.nvidia.com/nvidia/nemotron-ocr-v2) <br>
NGC 04/15/2026 via [https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo-microservices/containers/nemoretriever-ocr-v2](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo-microservices/containers/nemoretriever-ocr-v2) <br>

### Deployment Geography

Global

### Use Case

**Nemotron OCR v2** is designed for high-accuracy and high-speed extraction of textual information from images across multiple languages, making it ideal for powering multimodal retrieval systems, Retrieval-Augmented Generation (RAG) pipelines, and agentic applications that require seamless integration of visual and language understanding. Its robust multilingual performance and efficiency make it an excellent choice for next-generation AI systems that demand both precision and scalability across diverse real-world content.

### **Model Architecture**

**Architecture Type:** Hybrid detector-recognizer with document-level relational modeling

Nemotron OCR v2 is available in two variants:

- **v2_english** — Optimized for English-language OCR with word-level region handling.
- **v2_multilingual** — Supports English, Chinese (Simplified and Traditional), Japanese, Korean, and Russian with line-level region handling for multilingual documents.

Both variants share the same three-component architecture:

- **Text Detector:** Utilizes a RegNetX-8GF convolutional backbone for high-accuracy localization of text regions within images.
- **Text Recognizer:** Employs a pre-norm Transformer-based sequence recognizer to transcribe text from detected regions, supporting variable word and line lengths.
- **Relational Model:** Applies a multi-layer global relational module to predict logical groupings, reading order, and layout relationships across detected text elements.

All components are trained jointly in an end-to-end fashion, providing robust, scalable, and production-ready OCR for diverse document and scene images.

**Network Architecture**: RegNetX-8GF

#### Recognizer Comparison

The two variants share an identical detector and relational architecture but differ in recognizer capacity:

| Spec | v2_english | v2_multilingual |
|------|-----------|----------------|
| Transformer layers | 3 | 6 |
| Hidden dimension (`d_model`) | 256 | 512 |
| FFN width (`dim_feedforward`) | 1024 | 2048 |
| Attention heads | 8 | 8 |
| Max sequence length | 32 | 128 |
| Character set size | 855 | 14,244 |

#### Parameter Counts

**v2_english** (from `v2_english/`):

| Component         | Parameters  |
|-------------------|-------------|
| Detector          | 45,445,259  |
| Recognizer        | 6,130,657   |
| Relational model  | 2,255,419   |
| **Total**         | **53,831,335**  |

**v2_multilingual** (from `v2_multilingual/`):

| Component         | Parameters  |
|-------------------|-------------|
| Detector          | 45,445,259  |
| Recognizer        | 36,119,598  |
| Relational model  | 2,288,187   |
| **Total**         | **83,853,044**  |

### **Input**

| Property         | Value              |
|------------------|-------------------|
| Input Type & Format       | Image (RGB, PNG/JPEG, float32/uint8), aggregation level (word, sentence, or paragraph) |
| Input Parameters (Two-Dimensional)      | 3 x H x W (single image) or B x 3 x H x W (batch) |
| Input Range      | [0, 1] (float32) or [0, 255] (uint8, auto-converted) |
| Other Properties | Handles both single images and batches. Automatic multi-scale resizing for best accuracy. |

### **Output**

| Property        | Value              |
|-----------------|-------------------|
| Output Type     | Structured OCR results: a list of detected text regions (bounding boxes), recognized text, and confidence scores |
| Output Format   | Bounding boxes: tuple of floats, recognized text: string, confidence score: float |
| Output Parameters | Bounding boxes: One-Dimenional (1D) list of bounding box coordinates, recognized text: One-Dimenional (1D) list of strings, confidence score: One-Dimenional (1D) list of floats |
| Other Properties | Please see the sample output for an example of the model output |

### Sample output

```
ocr_boxes = [[[15.552736282348633, 43.141815185546875],
  [150.00149536132812, 43.141815185546875],
  [150.00149536132812, 56.845645904541016],
  [15.552736282348633, 56.845645904541016]],
 [[298.3145751953125, 44.43315124511719],
  [356.93585205078125, 44.43315124511719],
  [356.93585205078125, 57.34814453125],
  [298.3145751953125, 57.34814453125]],
 [[15.44686508178711, 13.67985725402832],
  [233.15859985351562, 13.67985725402832],
  [233.15859985351562, 27.376562118530273],
  [15.44686508178711, 27.376562118530273]],
 [[298.51727294921875, 14.268900871276855],
  [356.9850769042969, 14.268900871276855],
  [356.9850769042969, 27.790447235107422],
  [298.51727294921875, 27.790447235107422]]]

ocr_txts = ['The previous notice was dated',
 '22 April 2016',
 'The previous notice was given to the company on',
 '22 April 2016']

ocr_confs = [0.97730815, 0.98834222, 0.96804602, 0.98499225]
```

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA’s hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.


### Usage

#### Prerequisites

- **OS**: Linux amd64 with NVIDIA GPU
- **CUDA toolkit** with `nvcc` on `PATH`. The toolkit version must be compatible with
  the version of PyTorch you install (same major version). For example, if you install
  `torch` with CUDA 12.8 bindings, you need CUDA toolkit 12.x. Verify with
  `nvcc --version` and `nvidia-smi`.
- **Python**: 3.12 (the package requires `>=3.12,<3.13`)
- **Build tools** (for the C++ CUDA extension compiled at install time):
  - GCC/G++ with C++17 support
  - CUDA toolkit headers
  - OpenMP

#### Installation
The package includes a C++ CUDA extension that is compiled during installation.
Because the extension must be built against the **same PyTorch CUDA version** as
your system's CUDA toolkit, **install PyTorch first**, then install this package
with `--no-build-isolation` so it uses your existing PyTorch.

1. Clone the repository

- Make sure git-lfs is installed (https://git-lfs.com)
```
git lfs install
git clone https://huggingface.co/nvidia/nemotron-ocr-v2
```

2. Installation

##### With pip

- Create and activate a Python 3.12 environment
- Install PyTorch matching your CUDA toolkit (see https://pytorch.org/get-started/locally/):

```bash
# Example for CUDA 12.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

- Install the package:

```bash
cd nemotron-ocr
pip install --no-build-isolation -v .
```

- Verify the C++ extension loads:

```bash
python -c "from nemotron_ocr.inference.pipeline_v2 import NemotronOCRV2; print('OK')"
```

##### With docker

Run the example end-to-end without installing anything on the host (besides Docker, docker compose, and NVIDIA Container Toolkit):

- Ensure Docker can see your GPU:

```bash
docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.09-py3 nvidia-smi
```

- From the repo root, bring up the service to run the example (sample image `ocr-example-input-1.png` when present):

```bash
docker compose run --rm nemotron-ocr \
  bash -lc "python example.py ocr-example-input-1.png --merge-level paragraph"
```

This will:
- Build an image from the provided `Dockerfile` (based on `nvcr.io/nvidia/pytorch`)
- Mount the repo at `/workspace`
- Run `example.py` (downloads **v2 multilingual** from Hugging Face on first run unless you pass `--model-dir`)

Output is saved next to your input image as `<name>-annotated.<ext>` on the host.


3. Run the model using the following code.

Use `nemotron_ocr.inference.pipeline_v2.NemotronOCRV2`. With no arguments, checkpoints are downloaded from Hugging Face: **by default** the **v2 multilingual** bundle (`nvidia/nemotron-ocr-v2` / `v2_multilingual/`). Use `lang="en"` for the English v2 build (`nvidia/nemotron-ocr-v2` / `v2_english/`), or pass `model_dir` to load from disk (any complete checkpoint folder; `lang` is then ignored).

```python
from nemotron_ocr.inference.pipeline_v2 import NemotronOCRV2

# Default: Hugging Face v2 multilingual
ocr = NemotronOCRV2()

# English v2 (Hub, word-level)
ocr_en = NemotronOCRV2(lang="en")

# Multilingual v2 explicitly (same default as NemotronOCRV2())
# Uses the line-level variant.
ocr_multi = NemotronOCRV2(lang="multi")

# Local directory with detector.pth, recognizer.pth, relational.pth, charset.txt
ocr_local = NemotronOCRV2(model_dir="./v2_multilingual")

predictions = ocr("ocr-example-input-1.png")

for pred in predictions:
    print(
        f"  - Text: '{pred['text']}', "
        f"Confidence: {pred['confidence']:.2f}, "
        f"Bbox: [left={pred['left']:.4f}, upper={pred['upper']:.4f}, right={pred['right']:.4f}, lower={pred['lower']:.4f}]"
    )
```

#### Inference modes

```python
# Detector only — returns bounding boxes without text recognition.
# Loads only the detector (~37% less GPU memory, ~20% faster).
ocr_det = NemotronOCRV2(detector_only=True)
boxes = ocr_det("page.png")
# Each prediction has: confidence, left, right, upper, lower, quad

# Skip relational — returns per-word text without reading-order grouping.
# Skips the relational model (~35% less GPU memory, ~8% faster).
ocr_fast = NemotronOCRV2(skip_relational=True)
words = ocr_fast("page.png", merge_level="word")
# Each prediction has: text, confidence, left, right, upper, lower

# Profiling mode — enables per-phase CUDA-synced timing in the logs.
import logging
logging.basicConfig(level=logging.INFO)
ocr_profile = NemotronOCRV2(verbose_post=True)
```

**Constructor rules**

- You can choose model weights with either **`lang`** or **`model_dir`**.
- **`lang`** (keyword only, Hub download path):
  - `None`, `"multi"`, or `"multilingual"` -> **v2 multilingual** (default): `nvidia/nemotron-ocr-v2` / `v2_multilingual/`
  - `"en"` or `"english"` -> **v2 English**: `nvidia/nemotron-ocr-v2` / `v2_english/`
  - `"v1"` or `"legacy"` -> **v1 English-only** (backward compatibility): fetched from `nvidia/nemotron-ocr-v1` if not already cached locally
  - Both v2 variants (`v2_multilingual/` and `v2_english/`) are included in this repository.
- **`model_dir`** (local override): if it points to a complete local checkpoint directory (`detector.pth`, `recognizer.pth`, `relational.pth`, `charset.txt`), it takes precedence and **overrides `lang`**.
- If `model_dir` is provided but incomplete, loading falls back to Hub resolution via **`lang`** (defaulting to v2 multilingual when `lang` is `None`).

### Software Integration

**Runtime Engine(s):**
- PyTorch

**Supported Hardware Microarchitecture Compatibility:**
- NVIDIA Ampere
- NVIDIA Blackwell
- NVIDIA Hopper
- NVIDIA Lovelace

**Preferred/Supported Operating System(s):**
- Linux

## Model Version(s)

* **This repository:** Nemotron OCR v2 with both variants: `v2_english/` and `v2_multilingual/`.
* **Hugging Face Hub:** [nvidia/nemotron-ocr-v2](https://huggingface.co/nvidia/nemotron-ocr-v2).

## **Training and Evaluation Datasets:**

### **Training Dataset**

**Data Modality** 
* Image

**Image Training Data Size** 
* Approximately 12 million images

The model is trained on a large-scale, curated mix of real-world and synthetic OCR datasets spanning multiple languages, scripts, and document types.

**Real-world datasets (~680K images):** Natural scene text, multilingual scene text, arbitrary-shaped text, chart and infographic text, table images with bilingual annotations, and handwritten document pages. These cover diverse layouts, languages, and document types.

**Synthetic datasets (~11M+ images):** Rendered multilingual document pages in six languages (English, Japanese, Korean, Russian, Chinese Simplified, and Chinese Traditional) and synthetic historical document crops covering archaic characters with degradation effects.

**Data Collection Method by dataset:** Hybrid (Automated, Human, Synthetic)<br>
**Labeling Method by dataset:** Hybrid (Automated, Human, Synthetic)<br>
**Properties:** Includes scanned documents, natural scene images, charts, tables, infographics, handwritten documents, and synthetic rendered pages in multiple languages and scripts.

### **Evaluation Datasets**

Nemotron OCR v2 is evaluated on [OmniDocBench](https://github.com/opendatalab/OmniDocBench), a comprehensive document OCR benchmark covering English, Chinese, and mixed-language content across diverse document categories.

**Data Collection Method by dataset:** Hybrid (Automated, Human, Synthetic)<br>
**Labeling Method by dataset:** Hybrid (Automated, Human, Synthetic)<br>
**Properties:** Benchmarks include challenging scene images, documents with varied layouts, and multi-language data.

### **Evaluation Results**

Tables below are **reference metrics** from NVIDIA’s benchmark runs (OmniDocBench, SynthDoG). Reproducing them requires datasets and scripts that are **not** checked into this Hugging Face repository.

#### OmniDocBench

Normalized Edit Distance (NED) sample_avg on OmniDocBench (lower = better). Results follow OmniDocBench methodology (empty predictions skipped). All models evaluated in crop mode. Speed measured on a single A100 GPU.

| Model | pages/s | EN | ZH | Mixed | White | Single | Multi | Normal | Rotate90 | Rotate270 | Horizontal |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PaddleOCR v5 (server) | 1.2 | 0.027 | 0.037 | 0.041 | 0.031 | 0.035 | 0.064 | 0.031 | 0.116 | 0.897 | 0.027 |
| OpenOCR (server) | 1.5 | 0.024 | 0.033 | 0.049 | 0.027 | 0.034 | 0.061 | 0.028 | 0.042 | 0.761 | 0.034 |
| **Nemotron OCR v2 (multilingual)** | **34.7** | **0.048** | **0.072** | **0.142** | **0.061** | **0.049** | **0.117** | **0.062** | **0.109** | **0.332** | **0.372** |
| *Nemotron OCR v2 (EN)* | *40.7* | *0.038* | *0.830* | *0.437* | *0.348* | *0.282* | *0.572* | *0.353* | *0.232* | *0.827* | *0.893* |
| EasyOCR | 0.4 | 0.095 | 0.117 | 0.326 | 0.095 | 0.179 | 0.322 | 0.110 | 0.987 | 0.979 | 0.809 |
| *Nemotron OCR v1* | *39.3* | *0.038* | *0.876* | *0.436* | *0.472* | *0.434* | *0.715* | *0.482* | *0.358* | *0.871* | *0.979* |

Column key: **pages/s** is throughput using the v2 batched pipeline where measured; **EN** = English, **ZH** = Simplified Chinese, **Mixed** = English/Chinese mixed, **White/Single/Multi** = background type, **Normal/Rotate90/Rotate270/Horizontal** = text orientation.

#### [SynthDoG](https://github.com/clovaai/donut/tree/master/synthdog) Generated Benchmark Data

Normalized Edit Distance (NED) page_avg on [SynthDoG](https://github.com/clovaai/donut/tree/master/synthdog) generated benchmark data (lower = better):

| Language | PaddleOCR (base) | PaddleOCR (specialized) | OpenOCR (server) | Nemotron OCR v1 | *Nemotron OCR v2 (EN)* | **Nemotron OCR v2 (multilingual)** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| English | 0.117 | 0.096 | 0.105 | 0.078 | *0.079* | **0.069** |
| Japanese | 0.201 | 0.201 | 0.586 | 0.723 | *0.765* | **0.046** |
| Korean | 0.943 | 0.133 | 0.837 | 0.923 | *0.924* | **0.047** |
| Russian | 0.959 | 0.163 | 0.950 | 0.564 | *0.632* | **0.043** |
| Chinese (Simplified) | 0.054 | 0.054 | 0.061 | 0.784 | *0.819* | **0.035** |
| Chinese (Traditional) | 0.094 | 0.094 | 0.127 | 0.700 | *0.756* | **0.065** |

### **Detailed Performance Analysis**

The model demonstrates robust multilingual performance on complex layouts, noisy backgrounds, and challenging real-world scenes. Reading order and block detection are powered by the relational module, supporting downstream applications such as chart-to-text, table-to-text, and infographic-to-text extraction.

**Inference**<br>
**Acceleration Engine:** PyTorch<br>
**Supported Hardware:** H100 PCIe/SXM, A100 PCIe/SXM, L40S, L4, A10G, H200 NVL, B200, RTX PRO 6000 Blackwell Server Edition<br>

## Ethical Considerations

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>
The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment. <br>
Please make sure you have proper rights and permissions for all input image and video content; if image or video includes people, personal health information, or intellectual property, the image or video generated will not blur or maintain proportions of image subjects included. <br>
For more detailed information on ethical considerations for this model, please see the [Explainability](#explainability), [Bias](#bias), [Safety](#safety) & Security, and [Privacy](#privacy) sections below. <br>
Please report security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail).

## Bias

| Field | Response |
| ----- | ----- |
| Participation considerations from adversely impacted groups [protected classes](https://www.senate.ca.gov/content/protected-classes) in model design and testing | None |
| Measures taken to mitigate against unwanted bias | None |


## Explainability

| Field | Response |
| ----- | ----- |
| Intended Task/Domain: | Optical Character Recognition (OCR) with a focus on retrieval application and documents. |
| Model Type: | Hybrid neural network with convolutional detector, transformer recognizer, and document structure modeling. |
| Intended Users: | Developers and teams building AI-driven search applications, retrieval-augmented generation (RAG) workflows, multimodal agents, or document intelligence applications. It is ideal for those working with large collections of scanned or photographed documents, including PDFs, forms, and reports. |
| Output: | Structured OCR results, including detected bounding boxes, recognized text, and confidence scores. |
| Describe how the model works: | The model first detects text regions in the image, then transcribes recognized text, and finally analyzes document structure and reading order. Outputs structured, machine-readable results suitable for downstream search and analysis. |
| Name the adversely impacted groups this has been tested to deliver comparable outcomes regardless of: | Not Applicable |
| Technical Limitations & Mitigation: | Performance may vary across languages and scripts. |
| Verified to have met prescribed NVIDIA quality standards: | Yes |
| Performance Metrics: | Accuracy (e.g., character error rate), throughput, and latency. |
| Potential Known Risks: | The model may not always extract or transcribe all text with perfect accuracy, particularly in cases of poor image quality or highly stylized fonts. |
| Licensing & Terms of Use: | Use of this model is governed by [NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) and the use of the post-processing scripts are licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt). |


## Privacy

| Field | Response |
| ----- | ----- |
| Generatable or reverse engineerable personal data? | No |
| Personal data used to create this model? | None Known |
| How often is dataset reviewed? | The dataset is initially reviewed when added, and subsequent reviews are conducted as needed or in response to change requests. |
| Is there provenance for all datasets used in training? | Yes |
| Does data labeling (annotation, metadata) comply with privacy laws? | Yes |
| Is data compliant with data subject requests for data correction or removal, if such a request was made? | No, not possible with externally-sourced data. |
| Applicable Privacy Policy | https://www.nvidia.com/en-us/about-nvidia/privacy-policy/ |
| Was consent obtained for any personal data used? | Not Applicable |
| Was data from user interactions with the AI model (e.g. user input and prompts) used to train the model? | No |


## Safety

| Field | Response |
| ----- | ----- |
| Model Application Field(s): | Text recognition and structured OCR for multimodal retrieval. Inputs can include natural scene images, scanned documents, charts, tables, and infographics. |
| Use Case Restrictions: | Abide by [NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) and the use of the post-processing scripts are licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt). |
| Model and dataset restrictions: | The principle of least privilege (PoLP) is applied, limiting access for dataset generation and model development. Restrictions enforce dataset access only during training, and all dataset license constraints are adhered to. |
| Describe the life critical impact (if present): | Not applicable. |
