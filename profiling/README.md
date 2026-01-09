# LLM Inference Server GPU Profiler

NVIDIA Nsight Systems profiling for LLM inference servers (vLLM, Triton, TGI) on Kubernetes.

---

## Quick Start

### 1. Setup nsys Package on GPU Nodes

```bash
# Copy nsys package to cluster nodes
./scripts/setup-nsys-package.sh ../offline-packages/nsight-systems-cli-2025.6.1.deb bench
```

### 2. Run Profiling

```bash
# Profile single server
./scripts/run-profiler.sh vllm ./profiles
./scripts/run-profiler.sh triton ./profiles
./scripts/run-profiler.sh tgi ./profiles

# Profile all servers
./scripts/run-profiler.sh all ./profiles
```

### 3. View Profiles

```bash
nsys-ui ./profiles/vllm_gpu_timeline.nsys-rep
```

---

## Directory Structure

```
profiler_deployment/
├── README.md                    # This file
├── jobs/                        # Kubernetes Job definitions
│   ├── vllm-nsys-profiler.yaml  # vLLM profiler job
│   ├── triton-nsys-profiler.yaml# Triton profiler job
│   └── tgi-nsys-profiler.yaml   # TGI profiler job
├── scripts/                     # Helper scripts
│   ├── setup-nsys-package.sh    # Copy nsys to cluster nodes
│   ├── run-profiler.sh          # Automated profiling workflow
│   └── extract-profiles.sh      # Extract profiles from pods
└── docs/                        # Documentation
    ├── PROFILING_METHODOLOGY.md # Detailed methodology
    └── BEST_PRACTICES.md        # Best practices guide
```

---

## Prerequisites

1. **Kubernetes Cluster** with GPU nodes
2. **kubectl** configured with cluster access
3. **nsys .deb package** (Nsight Systems CLI)
   - Download from: https://developer.nvidia.com/nsight-systems
   - Version: 2025.6.1 or later

4. **Container Images** for inference servers:
   - vLLM: `fra.ocir.io/frntrd2vyxvi/models:llm-vllm-offline-v1`
   - Triton: `fra.ocir.io/frntrd2vyxvi/models:llm-triton-offline-v1`
   - TGI: `fra.ocir.io/frntrd2vyxvi/models:llm-tgi-offline-v1`

5. **Namespace**: `bench` with service account `bench-sa`

---

## Manual Deployment

### Step 1: Scale Down Existing Deployment

```bash
kubectl scale deployment vllm-server -n bench --replicas=0
```

### Step 2: Deploy Profiler Job

```bash
kubectl apply -f jobs/vllm-nsys-profiler.yaml
```

### Step 3: Monitor Progress

```bash
kubectl logs -f job/vllm-nsys-profiler -n bench
```

### Step 4: Extract Profile

```bash
POD=$(kubectl get pods -n bench -l job-name=vllm-nsys-profiler -o jsonpath='{.items[0].metadata.name}')
kubectl cp bench/$POD:/results/vllm_gpu_timeline.nsys-rep ./vllm_gpu_timeline.nsys-rep
```

### Step 5: Cleanup and Restore

```bash
kubectl delete job vllm-nsys-profiler -n bench
kubectl scale deployment vllm-server -n bench --replicas=1
```

---

## Key Concept: Why Server Must Start Under nsys

```
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│   WRONG: Attach to running server                           │
│   ┌──────────┐     ┌──────────┐                             │
│   │ Server   │────▶│ nsys     │  = No CUDA kernel data      │
│   │ (running)│     │ (attach) │                             │
│   └──────────┘     └──────────┘                             │
│                                                              │
│   RIGHT: Start server under nsys                            │
│   ┌──────────┐     ┌──────────┐                             │
│   │ nsys     │────▶│ Server   │  = Full CUDA timeline       │
│   │ (parent) │     │ (child)  │                             │
│   └──────────┘     └──────────┘                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

CUDA profiling hooks must be injected at CUDA initialization time. Once a process has initialized CUDA, nsys cannot retroactively trace GPU operations.

---

## Expected Profile Sizes

| Server | Profile Size | Quality |
|--------|--------------|---------|
| vLLM | 15-25 MB | Full inference |
| Triton | 5-15 MB | Model loading + inference |
| TGI | 10-20 MB | Full inference |

Profiles under 1 MB typically indicate missing CUDA data.

---

## nsys Command Reference

```bash
nsys profile \
    --output=/results/profile_name \
    --trace=cuda,nvtx,cudnn,cublas,osrt \
    --cuda-memory-usage=true \
    --cudabacktrace=kernel \
    --force-overwrite=true \
    --duration=240 \
    --sample=process-tree \
    --backtrace=dwarf \
    <server-command>
```

| Option | Purpose |
|--------|---------|
| `--trace` | What to capture (cuda, nvtx, cudnn, cublas, osrt) |
| `--cuda-memory-usage` | Track GPU memory allocations |
| `--cudabacktrace` | Capture kernel launch call stacks |
| `--duration` | Maximum profiling time (seconds) |
| `--sample=process-tree` | Trace all child processes |

---

## Viewing Profiles

### GUI (Recommended)

```bash
# Install Nsight Systems GUI from NVIDIA website
nsys-ui profile.nsys-rep
```

### Command Line

```bash
# Summary statistics
nsys stats profile.nsys-rep

# GPU kernel report
nsys stats --report gputrace profile.nsys-rep

# Export to JSON
nsys export --type=json -o profile.json profile.nsys-rep
```

---

## Troubleshooting

### nsys: command not found

Use full path: `/opt/nvidia/nsight-systems-cli/2025.6.1/bin/nsys`

### Profile is small (< 1 MB)

Server wasn't started under nsys. Ensure the inference server command is passed directly to `nsys profile`.

### dpkg fails with missing dependencies

Use `dpkg --force-depends -i nsight-systems-cli.deb`

### Model not loading

- Use PVC with pre-cached model
- Ensure network access for HuggingFace Hub
- Increase `--duration` value

---

## Documentation

- [Profiling Methodology](docs/PROFILING_METHODOLOGY.md) - Detailed workflow
- [Best Practices](docs/BEST_PRACTICES.md) - Tips and recommendations

---

## Results from This Framework

Successfully captured profiles:

| Profile | Size | Contains |
|---------|------|----------|
| vllm_gpu_timeline.nsys-rep | 19 MB | Full CUDA kernels during inference |
| triton_full_gpu_timeline.nsys-rep | 6.3 MB | Model loading + vLLM backend |
| tgi_full_gpu_timeline.nsys-rep | 886 KB | Startup (model cache unavailable) |

---

*LLM Inference Benchmarking Framework - GPU Profiling Module*
