# LLM Inference Benchmark Client with Nsight Systems
# For profiling and benchmarking vLLM, Triton, TGI
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda-12.1 \
    PATH=/usr/local/cuda-12.1/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip python3-dev \
    curl wget jq vim netcat-openbsd \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python setup
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    python -m pip install --upgrade pip==23.3.1

# Python dependencies (all pre-installed, no runtime downloads)
RUN pip install --no-cache-dir \
    numpy==1.24.3 \
    matplotlib==3.7.2 \
    pyyaml==6.0 \
    requests==2.31.0 \
    pandas==2.0.3

# ========================================
# NSIGHT SYSTEMS INSTALLATION (OFFLINE)
# ========================================
# Copy from offline-packages (preferred - no network required)
COPY docker/bench-client/offline-packages/nsight-systems-cli-2025.6.1.deb /tmp/nsight-systems.deb
RUN dpkg -i /tmp/nsight-systems.deb || true && \
    apt-get update && apt-get install -f -y && \
    rm /tmp/nsight-systems.deb && \
    rm -rf /var/lib/apt/lists/*

# Add nsys to PATH (adjust version as needed)
ENV PATH=/opt/nvidia/nsight-systems/2024.6.1/bin:/opt/nvidia/nsight-systems/2025.6.1/bin:$PATH

# Verify nsys installation
RUN nsys --version || echo "nsys version check (may show warning on non-GPU host)"

# ========================================
# GPU VERIFICATION SCRIPT
# ========================================
COPY docker/bench-client/verify-gpu.sh /usr/local/bin/verify-gpu.sh
RUN chmod +x /usr/local/bin/verify-gpu.sh

# ========================================
# APPLICATION CODE
# ========================================
WORKDIR /app

# Copy benchmark client code
COPY bench-client/ /app/bench-client/
COPY nsys/ /app/nsys/
COPY configs/ /app/configs/

# Make scripts executable
RUN chmod +x /app/nsys/scripts/*.sh 2>/dev/null || true

# Create results directories
RUN mkdir -p /results/profiles /results/benchmarks /results/plots && \
    chmod -R 777 /results

# NCCL settings for OKE (lessons learned from LLM-training project)
# NOTE: NCCL_SOCKET_IFNAME set via K8s env var (ens3 for OKE)
ENV NCCL_DEBUG=INFO \
    NCCL_IB_DISABLE=1 \
    NCCL_P2P_DISABLE=0 \
    NCCL_ASYNC_ERROR_HANDLING=1

# Environment for profiling
# NOTE: --gpu-metrics-devices removed from nsys (requires privileged mode on OKE)
# NOTE: --cpuctxsw must be process-tree, system-wide, or none (NOT true/false)
ENV NSYS_OUTPUT_DIR=/results/profiles \
    NSYS_TRACE_OPTIONS="cuda,nvtx,osrt,cudnn,cublas"

# Default command
CMD ["python3", "/app/bench-client/inference_client.py", "--help"]
