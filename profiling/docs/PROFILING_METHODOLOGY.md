# Nsight Systems GPU Profiling Methodology

## LLM Inference Server Profiling on Kubernetes

**Environment**: Oracle Kubernetes Engine (OKE)
**GPU**: NVIDIA A10 (24GB VRAM)
**Profiler**: NVIDIA Nsight Systems 2025.6.1

---

## Overview

This document describes the methodology used to capture GPU timeline profiles for LLM inference servers (vLLM, Triton, TGI) running on Kubernetes with NVIDIA GPUs.

---

## Key Insight: Why Server Must Start Under nsys

**Critical Requirement**: The inference server process MUST be started wrapped with `nsys profile` to capture CUDA kernel data.

### Why This Matters

```
┌─────────────────────────────────────────────────────────────────┐
│                    CUDA Profiling Architecture                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   WRONG: Attach to running process                              │
│   ┌──────────────┐     ┌──────────────┐                         │
│   │ Server       │────▶│ nsys attach  │  = No CUDA data         │
│   │ (already     │     │ (too late)   │    Only OS/CPU traces   │
│   │  running)    │     └──────────────┘                         │
│   └──────────────┘                                              │
│                                                                  │
│   CORRECT: Start server under nsys                              │
│   ┌──────────────┐     ┌──────────────┐                         │
│   │ nsys profile │────▶│ Server       │  = Full CUDA data       │
│   │ (wrapper)    │     │ (child proc) │    Kernels, memory,     │
│   └──────────────┘     └──────────────┘    timeline             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Reason**: CUDA profiling requires instrumentation hooks to be injected at CUDA initialization time. Once a process has already initialized CUDA, nsys cannot retroactively inject these hooks.

---

## Profiling Workflow

### Step 1: Prepare Infrastructure

```
┌─────────────────────────────────────────────────────────────────┐
│                     Preparation Phase                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Store nsys package on cluster nodes                         │
│     └── Copy .deb to /tmp/ on GPU nodes                         │
│                                                                  │
│  2. Scale down existing deployment (if any)                     │
│     └── kubectl scale deployment <name> --replicas=0            │
│                                                                  │
│  3. Prepare model cache (optional but recommended)              │
│     └── Use PVC with pre-downloaded model weights               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Step 2: Deploy Profiling Job

```
┌─────────────────────────────────────────────────────────────────┐
│                     Profiling Job Structure                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Kubernetes Job                                                  │
│  ├── Init Container (optional)                                  │
│  │   └── Setup model repository / config files                  │
│  │                                                               │
│  └── Main Container                                              │
│      ├── 1. Install nsys from package                           │
│      ├── 2. Start server wrapped with nsys profile              │
│      ├── 3. Wait for server ready (health endpoint)             │
│      ├── 4. Send inference requests                             │
│      ├── 5. Wait for profiling to complete                      │
│      └── 6. Keep pod alive for file extraction                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Step 3: Extract Profile

```bash
# Copy profile from pod to local machine
kubectl cp bench/<pod-name>:/results/profile.nsys-rep ./profile.nsys-rep

# Verify file size (should be several MB for good profile)
ls -lh profile.nsys-rep
```

### Step 4: Analyze Profile

```bash
# GUI analysis (recommended)
nsys-ui profile.nsys-rep

# Command-line statistics
nsys stats profile.nsys-rep

# Export to other formats
nsys export --type=json -o profile.json profile.nsys-rep
```

---

## nsys Profile Command Options

### Recommended Command Template

```bash
nsys profile \
    --output=/results/<server>_gpu_timeline \
    --trace=cuda,nvtx,cudnn,cublas,osrt \
    --cuda-memory-usage=true \
    --cudabacktrace=kernel \
    --force-overwrite=true \
    --duration=240 \
    --sample=process-tree \
    --backtrace=dwarf \
    <server-command>
```

### Option Explanations

| Option | Purpose | Recommended Value |
|--------|---------|-------------------|
| `--output` | Output file path (without extension) | `/results/<name>` |
| `--trace` | What to trace | `cuda,nvtx,cudnn,cublas,osrt` |
| `--cuda-memory-usage` | Track GPU memory allocations | `true` |
| `--cudabacktrace` | Capture call stacks for CUDA APIs | `kernel` or `all` |
| `--force-overwrite` | Overwrite existing files | `true` |
| `--duration` | Max profiling duration (seconds) | `180-300` |
| `--sample` | Process sampling mode | `process-tree` |
| `--backtrace` | CPU backtrace method | `dwarf` |

### Trace Types Explained

| Trace Type | What It Captures |
|------------|------------------|
| `cuda` | CUDA API calls, kernel launches, memory operations |
| `nvtx` | NVIDIA Tools Extension markers (if used by application) |
| `cudnn` | cuDNN library calls (convolutions, etc.) |
| `cublas` | cuBLAS library calls (matrix operations) |
| `osrt` | OS runtime (threads, I/O, synchronization) |

---

## Server-Specific Considerations

### vLLM

```bash
nsys profile \
    --output=/results/vllm_gpu_timeline \
    --trace=cuda,nvtx,cudnn,cublas,osrt \
    --cuda-memory-usage=true \
    --cudabacktrace=kernel \
    --duration=240 \
    --sample=process-tree \
    python3 -m vllm.entrypoints.openai.api_server \
        --model mistralai/Mistral-7B-Instruct-v0.2 \
        --host 0.0.0.0 --port 8000 \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.85
```

**Notes**:
- vLLM uses PagedAttention - profiles show efficient memory management
- Model loading takes 2-3 minutes before inference starts
- Use `--gpu-memory-utilization 0.85` to leave room for profiling overhead

### NVIDIA Triton (with vLLM backend)

```bash
nsys profile \
    --output=/results/triton_gpu_timeline \
    --trace=cuda,nvtx,cudnn,cublas,osrt \
    --cuda-memory-usage=true \
    --cudabacktrace=kernel \
    --duration=240 \
    --sample=process-tree \
    tritonserver \
        --model-repository=/model-repository \
        --http-port=8000 \
        --grpc-port=8001 \
        --strict-model-config=false
```

**Notes**:
- Requires model repository with config.pbtxt
- vLLM backend loads inside Triton process
- Use init container to setup model repository

### HuggingFace TGI

```bash
nsys profile \
    --output=/results/tgi_gpu_timeline \
    --trace=cuda,nvtx,cudnn,cublas,osrt \
    --cuda-memory-usage=true \
    --cudabacktrace=kernel \
    --duration=240 \
    --sample=process-tree \
    text-generation-launcher \
        --model-id mistralai/Mistral-7B-Instruct-v0.2 \
        --port 8000 \
        --max-input-length 4096 \
        --max-total-tokens 8192 \
        --dtype float16
```

**Notes**:
- TGI uses Flash Attention - profiles show fused kernels
- Model must be pre-cached or have network access
- Use `--dtype float16` for A10 GPUs

---

## Troubleshooting

### Problem: nsys command not found after installation

**Solution**: Use full path to nsys binary
```bash
/opt/nvidia/nsight-systems-cli/2025.6.1/bin/nsys profile ...
```

### Problem: Profile file is small (<1MB) with no CUDA data

**Cause**: Server was already running or nsys couldn't trace CUDA
**Solution**: Ensure server starts as child of nsys process

### Problem: dpkg installation fails with missing dependencies

**Solution**: Use --force-depends flag
```bash
dpkg --force-depends -i nsight-systems-cli.deb
```

### Problem: Model not loading (timeout)

**Solutions**:
1. Increase wait timeout in script
2. Use PVC with pre-cached model
3. Ensure network access for model download
4. Use offline container image with pre-baked model

### Problem: Profile file corrupted after kubectl cp

**Solution**: Verify file sizes match
```bash
# In pod
ls -la /results/*.nsys-rep

# After copy
ls -la *.nsys-rep
```

---

## Best Practices

### 1. Duration Planning

```
┌─────────────────────────────────────────────────────────────────┐
│                    Profiling Timeline                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  0s          60s         120s        180s        240s           │
│  │           │           │           │           │              │
│  ├───────────┼───────────┼───────────┼───────────┤              │
│  │  Model    │  Server   │ Inference │  Buffer   │              │
│  │  Loading  │  Ready    │  Requests │  Time     │              │
│  │  (warm)   │  (idle)   │  (active) │           │              │
│  └───────────┴───────────┴───────────┴───────────┘              │
│                                                                  │
│  Recommended: --duration=240 (4 minutes)                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Inference Load Generation

Send enough requests to capture meaningful GPU activity:
- **Minimum**: 10 requests
- **Recommended**: 20-50 requests
- **Token count**: 100-200 tokens per request
- **Spacing**: 2-3 seconds between requests

### 3. Resource Allocation

```yaml
resources:
  requests:
    memory: "32Gi"
    cpu: "8"
    nvidia.com/gpu: "1"
  limits:
    memory: "48Gi"
    cpu: "16"
    nvidia.com/gpu: "1"
```

### 4. Shared Memory for PyTorch

```yaml
volumes:
- name: dshm
  emptyDir:
    medium: Memory
    sizeLimit: "16Gi"
volumeMounts:
- name: dshm
  mountPath: /dev/shm
```

---

## Expected Profile Sizes

| Server | Good Profile Size | Contains |
|--------|-------------------|----------|
| vLLM | 15-25 MB | Full inference with CUDA kernels |
| Triton | 5-15 MB | Model loading + inference |
| TGI | 10-20 MB | Full inference with Flash Attention |

Profiles under 1MB typically indicate profiling issues (no CUDA data captured).

---

## What to Look For in Profiles

### In nsys-ui Timeline View

1. **CUDA API Row**: cudaLaunchKernel, cudaMemcpy calls
2. **GPU Kernels Row**: Individual kernel executions (attention, GEMM)
3. **Memory Row**: Allocation/deallocation patterns
4. **CPU Threads**: Python/server thread activity

### Key Metrics

| Metric | What It Tells You |
|--------|-------------------|
| Kernel duration | Time spent in GPU compute |
| Memory transfers | CPU-GPU data movement overhead |
| Kernel occupancy | GPU utilization efficiency |
| API call gaps | Potential CPU bottlenecks |

---

*Document generated as part of LLM Inference Benchmarking Framework*
