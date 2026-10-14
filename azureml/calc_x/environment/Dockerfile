# Training image for the Calc-X experiment: Python 3.12 + the GPU stack verified locally (torch 2.11 cu130, vLLM 0.22,
# verl 0.8.0, flash-attn 2.8.3) + the Calc-X agent's deps + MLflow. Python deps are baked in, pinned in requirements.txt.
#
# CUDA user-space libraries come from the pip wheels, so a plain Ubuntu base is enough. The host only needs an NVIDIA
# driver that supports CUDA 13.0 (>= 580).
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
# build-essential: vLLM's flashinfer kernels are JIT-compiled at runtime. curl: health checks in run_job.sh.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates curl git procps \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.22 /uv /uvx /bin/
# Python lives outside /root so the image also works when run as a non-root user (see `make local`).
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    VIRTUAL_ENV=/opt/venv
RUN uv venv --python 3.12.15 /opt/venv

COPY requirements.txt /tmp/requirements.txt
RUN uv pip install --no-deps -r /tmp/requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match \
    && rm /tmp/requirements.txt

# flash-attn has no prebuilt wheel for torch 2.11 + CUDA 13.0, and compiling it here would take hours on AzureML's
# image builder. `make flash-attn-wheel` compiles it once (sm80 + sm90) into wheels/.
COPY wheels/ /tmp/wheels/
RUN uv pip install --no-deps /tmp/wheels/flash_attn-*.whl && rm -rf /tmp/wheels

# CUDA 13.0 toolchain from the nvidia-* wheels, for flashinfer's runtime JIT (same layout as scripts/setup_verl.sh).
ENV CUDA_HOME=/opt/venv/lib/python3.12/site-packages/nvidia/cu13
RUN ln -sf $CUDA_HOME/lib/libcudart.so.13 $CUDA_HOME/lib/libcudart.so
ENV PATH=/opt/venv/bin:$CUDA_HOME/bin:$PATH \
    LIBRARY_PATH=$CUDA_HOME/lib \
    CPATH=$CUDA_HOME/include \
    PYTHONUNBUFFERED=1
