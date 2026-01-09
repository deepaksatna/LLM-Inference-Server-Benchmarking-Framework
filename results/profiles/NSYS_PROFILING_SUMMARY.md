# Nsight Systems GPU Timeline Profiling Summary

## Date: January 8, 2026

---

## Generated Profile Files

| File | Size | Server | Status | Contains CUDA Data |
|------|------|--------|--------|-------------------|
| `vllm_gpu_timeline.nsys-rep` | **19 MB** | vLLM | **Complete** | **Yes - Full inference** |
| `triton_full_gpu_timeline.nsys-rep` | **6.3 MB** | Triton | **Model Loading** | **Yes - Startup/Load** |
| `tgi_full_gpu_timeline.nsys-rep` | 886 KB | TGI | Startup | Partial |
| `triton_gpu_timeline.nsys-rep` | 89 KB | Triton | Client-side | No |
| `tgi_gpu_timeline.nsys-rep` | 64 KB | TGI | Client-side | No |
| `tgi_gpu_timeline_attempt.nsys-rep` | 880 KB | TGI | Startup only | Partial |

### Best Profiles for Timeline Analysis

1. **vLLM (19MB)** - Complete GPU timeline with CUDA kernels during active inference
2. **Triton (6.3MB)** - GPU timeline during model loading and vLLM backend startup

---

## vLLM Profile (Best Profile)

### Profile Details
- **Size**: 19 MB
- **Duration**: ~45 seconds of profiling
- **Inference Requests**: 10 requests @ 200 tokens each
- **Contents**: Full CUDA kernel traces, GPU memory operations, timeline data

### How to View
```bash
# Open in Nsight Systems GUI (recommended)
nsys-ui vllm_gpu_timeline.nsys-rep

# Command-line stats
nsys stats vllm_gpu_timeline.nsys-rep

# Export to JSON
nsys export --type=json -o vllm_profile.json vllm_gpu_timeline.nsys-rep
```

### Captured Data Includes
- CUDA kernel execution timeline
- GPU memory allocations/deallocations
- cuBLAS/cuDNN operations
- Memory bandwidth utilization
- CPU-GPU synchronization points
- OS runtime activity (poll, recv, send)

---

## Why vLLM Profile is Best

The vLLM profile successfully captured GPU kernel data because:

1. **Server started under nsys**: The vLLM server process was launched wrapped with `nsys profile`, enabling CUDA tracing from the start
2. **Process tree tracing**: Used `--sample=process-tree` to capture all child processes
3. **Full CUDA instrumentation**: `--trace=cuda,nvtx,cudnn,cublas` captured all GPU operations

### Profiling Command Used
```bash
nsys profile \
    --output=/results/vllm_gpu_timeline \
    --trace=cuda,nvtx,cudnn,cublas,osrt \
    --cuda-memory-usage=true \
    --cudabacktrace=kernel \
    --duration=240 \
    --sample=process-tree \
    --backtrace=dwarf \
    python3 -m vllm.entrypoints.openai.api_server \
        --model mistralai/Mistral-7B-Instruct-v0.2 \
        --host 0.0.0.0 --port 8000 \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.85
```

---

## Triton and TGI Limitations

### Why Limited Data
The Triton and TGI profiles have limited CUDA data because:

1. **Server not started under nsys**: The servers were already running when profiling attempted
2. **System-wide CUDA tracing limitations**: nsys cannot inject CUDA tracing into already-running processes
3. **Requires process restart**: To capture full GPU timelines, the inference server must be started wrapped with nsys

### What Was Captured
- OS runtime activity (network I/O, file operations)
- CPU thread scheduling
- Process timing

### To Get Full Triton/TGI Profiles
Follow the same approach as vLLM:

1. Scale down deployment: `kubectl scale deployment triton-server -n bench --replicas=0`
2. Create profiling pod with server started under nsys
3. Wait for server ready, send inference requests
4. Copy the generated .nsys-rep file

---

## Timeline Visualization Guide

### Opening in Nsight Systems GUI

1. Install Nsight Systems GUI (available for Windows, Linux, macOS)
2. Open File → Open → Select `vllm_gpu_timeline.nsys-rep`
3. Navigate the timeline view

### Key Areas to Examine

1. **CUDA API Timeline**: Shows cudaLaunch, cudaMemcpy calls
2. **GPU Kernels**: Individual kernel executions (attention, GEMM, etc.)
3. **Memory Operations**: GPU memory allocations/transfers
4. **Thread Activity**: CPU thread scheduling during inference

### Recommended Views
- Timeline View: See kernel execution patterns
- Statistics View: Aggregate kernel timing
- Events Table: Detailed per-kernel metrics

---

## Methodology

### Infrastructure
- **Kubernetes**: Oracle Kubernetes Engine (OKE)
- **GPU**: NVIDIA A10 (24GB VRAM)
- **Model**: Mistral-7B-Instruct-v0.2 (FP16)

### Profiling Process
1. Created standalone profiling pod for each server
2. Installed nsys 2025.6.1 from offline package
3. Started server wrapped with nsys profile command
4. Waited for server ready (health endpoint)
5. Sent 10 inference requests (200 tokens each)
6. Waited for profile generation
7. Copied .nsys-rep file to local machine

### Files Location
```
results/profiles/20260108_nsys_timelines/
├── vllm_gpu_timeline.nsys-rep     # BEST - Full GPU timeline
├── triton_gpu_timeline.nsys-rep   # Partial - Client-side only
├── tgi_gpu_timeline.nsys-rep      # Partial - Client-side only
├── tgi_gpu_timeline_attempt.nsys-rep  # Startup attempt
└── NSYS_PROFILING_SUMMARY.md      # This file
```

---

## Recommendations

### For Best Results
1. **Always start server under nsys** to capture full CUDA traces
2. **Use adequate duration** (240+ seconds) for model loading + inference
3. **Generate inference load** after server is ready
4. **Use offline nsys package** for consistent versions across pods

### For Production Profiling
1. Create dedicated profiling Jobs/Pods
2. Use init containers for nsys installation
3. Mount model caches to avoid download delays
4. Consider longer profiling duration for comprehensive data

---

*Generated by LLM Inference Benchmarking Framework v1.0*
