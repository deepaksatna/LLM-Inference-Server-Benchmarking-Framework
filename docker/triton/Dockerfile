# Triton Inference Server with vLLM backend + pre-downloaded Mistral model
# Fully offline - no runtime downloads required
FROM nvcr.io/nvidia/tritonserver:24.08-vllm-python-py3

# NCCL settings for OKE (lessons learned from LLM-training project)
# NOTE: Do NOT set NCCL_SOCKET_IFNAME here - set via K8s env var (ens3 for OKE)
ENV NCCL_DEBUG=INFO \
    NCCL_IB_DISABLE=1 \
    NCCL_P2P_DISABLE=0 \
    NCCL_ASYNC_ERROR_HANDLING=1 \
    TORCH_NCCL_BLOCKING_WAIT=1

# HF token for build-time download
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Copy and run download script (BEFORE setting offline mode)
COPY docker/triton/download_model.py /tmp/download_model.py
RUN python3 /tmp/download_model.py && rm /tmp/download_model.py

# Clear token
ENV HF_TOKEN=""

# NOW set offline mode (AFTER download complete)
ENV HF_DATASETS_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    DISABLE_TELEMETRY=1

# Create Triton model repository structure
RUN mkdir -p /models/mistral/1

# Copy Triton config files
COPY docker/triton/config.pbtxt /models/mistral/config.pbtxt
COPY docker/triton/model.json /models/mistral/1/model.json

# Verify model repository structure
RUN echo "Model repository structure:" && \
    find /models -type f && \
    echo "" && \
    cat /models/mistral/config.pbtxt

# Default command
CMD ["tritonserver", \
     "--model-repository=/models", \
     "--http-port=8000", \
     "--grpc-port=8001", \
     "--metrics-port=8002", \
     "--log-verbose=1"]
