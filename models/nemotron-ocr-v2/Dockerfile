FROM nvcr.io/nvidia/pytorch:25.09-py3

ARG TARGETARCH

ARG TORCH_CUDA_ARCH_LIST
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -U pip hatchling "setuptools>=68" --root-user-action ignore

COPY nemotron-ocr /workspace/nemotron-ocr
WORKDIR /workspace/nemotron-ocr

# Ensure no prebuilt binaries/artifacts from the host are present
RUN rm -f src/nemotron_ocr_cpp/*.so || true \
    && rm -rf build/ dist/

RUN --mount=type=cache,target=/root/.cache/pip \
    BUILD_CPP_FORCE=1 ARCH=${TARGETARCH} pip install -v . --no-build-isolation --root-user-action ignore

WORKDIR /workspace
