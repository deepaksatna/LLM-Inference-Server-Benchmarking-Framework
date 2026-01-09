# Nsight Systems Profiling Best Practices

## For LLM Inference Servers on Kubernetes

---

## The Golden Rule

> **Start the inference server as a child process of nsys profile**

This is the single most important requirement for capturing CUDA kernel data. If you don't follow this rule, your profiles will only contain OS-level traces without any GPU kernel information.

```
WRONG:  Server running → nsys attaches → No CUDA data
RIGHT:  nsys starts → Server runs as child → Full CUDA data
```

---

## Best Practices by Category

### 1. nsys Command Configuration

**Optimal Command for LLM Inference:**

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

**Parameter Recommendations:**

| Parameter | Recommended | Why |
|-----------|-------------|-----|
| `--trace` | `cuda,nvtx,cudnn,cublas,osrt` | Captures all GPU operations |
| `--cuda-memory-usage` | `true` | Essential for memory analysis |
| `--cudabacktrace` | `kernel` | Shows which code launched each kernel |
| `--duration` | 180-300s | Allow time for model loading + inference |
| `--sample` | `process-tree` | Captures all child processes |

### 2. Duration Planning

**Timeline Breakdown:**

```
Model Loading: 60-180s (depends on model size and cache)
Server Ready:  10-30s (warmup)
Inference:     60-120s (your test requests)
Buffer:        30-60s (safety margin)
─────────────────────────────────────────────
Total:         180-300s recommended
```

**Model-Specific Durations:**

| Model Size | Recommended Duration |
|------------|---------------------|
| 7B parameters | 180s |
| 13B parameters | 240s |
| 30B+ parameters | 300s+ |

### 3. Inference Load Generation

**Minimum Requirements:**
- 10+ requests (more = better statistical significance)
- 100-200 tokens per request (captures full generation cycle)
- 2-3 second spacing (allows observation of individual requests)

**Recommended Load Pattern:**

```bash
for i in $(seq 1 20); do
    curl -s http://localhost:8000/v1/completions \
        -d '{"prompt":"...","max_tokens":200}'
    sleep 2
done
```

**Why This Matters:**
- Too few requests = incomplete picture of steady-state behavior
- Too many concurrent = may hide individual kernel patterns
- Proper spacing = clear request boundaries in timeline

### 4. Kubernetes Job Design

**Essential Pod Spec Elements:**

```yaml
spec:
  # Prevent restart loops
  restartPolicy: Never

  # Required for GPU access
  tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule

  # Target GPU nodes
  nodeSelector:
    nvidia.com/gpu: "true"

  # Adequate resources
  resources:
    requests:
      memory: "32Gi"
      cpu: "8"
      nvidia.com/gpu: "1"
    limits:
      memory: "48Gi"
      cpu: "16"
      nvidia.com/gpu: "1"

  # Shared memory for PyTorch
  volumes:
  - name: dshm
    emptyDir:
      medium: Memory
      sizeLimit: "16Gi"
```

**Why Each Element:**

| Element | Purpose |
|---------|---------|
| `restartPolicy: Never` | Prevent restart after profiling completes |
| `tolerations` | Allow scheduling on GPU nodes with taints |
| `nodeSelector` | Ensure pod lands on GPU node |
| `resources.limits.gpu` | Actually allocate GPU to pod |
| `dshm volume` | PyTorch requires large shared memory |

### 5. nsys Package Installation

**Handling Missing Dependencies:**

```bash
# Standard installation
dpkg -i nsight-systems-cli.deb

# If dependencies missing (common in minimal containers)
dpkg --force-depends -i nsight-systems-cli.deb
```

**Use Full Path:**

```bash
# Don't rely on PATH
NSYS="/opt/nvidia/nsight-systems-cli/2025.6.1/bin/nsys"
$NSYS profile ...

# NOT this (may fail in scripts)
export PATH="/opt/nvidia/nsight-systems-cli/2025.6.1/bin:$PATH"
nsys profile ...
```

### 6. Profile Extraction

**Verify Before Extraction:**

```bash
# Check file exists and has reasonable size
kubectl exec -n bench $POD -- ls -lh /results/*.nsys-rep

# Good: 15-25 MB for LLM inference
# Bad: < 1 MB (likely no CUDA data)
```

**Extract with Verification:**

```bash
# Extract
kubectl cp bench/$POD:/results/profile.nsys-rep ./profile.nsys-rep

# Verify
ls -lh ./profile.nsys-rep
# Should match size shown in pod
```

### 7. Model Caching

**Options for Model Availability:**

1. **Pre-baked Container Image** (Fastest)
   - Model weights included in Docker image
   - No download time during profiling

2. **Persistent Volume Claim** (Recommended)
   - Model cached on PVC
   - Survives pod restarts

3. **Network Download** (Slowest)
   - Downloads from HuggingFace Hub
   - Requires network access
   - Add 10-20 minutes to duration

**PVC Example:**

```yaml
volumes:
- name: model-cache
  persistentVolumeClaim:
    claimName: huggingface-models-pvc
```

---

## Common Mistakes and Fixes

### Mistake 1: Profiling Already-Running Server

**Wrong:**
```bash
# Server already running
nsys profile --trace=cuda sleep 60  # Won't capture server's CUDA
```

**Right:**
```bash
# Start server under nsys
nsys profile --trace=cuda python -m vllm.entrypoints.openai.api_server ...
```

### Mistake 2: Too Short Duration

**Wrong:**
```bash
--duration=60  # Not enough for model loading
```

**Right:**
```bash
--duration=240  # 4 minutes for 7B model
```

### Mistake 3: Missing Trace Types

**Wrong:**
```bash
--trace=cuda  # Missing important traces
```

**Right:**
```bash
--trace=cuda,nvtx,cudnn,cublas,osrt  # Complete picture
```

### Mistake 4: Forgetting Shared Memory

**Wrong:**
```yaml
# No dshm volume - PyTorch may crash
```

**Right:**
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

### Mistake 5: Not Waiting for Server Ready

**Wrong:**
```bash
# Start server and immediately send requests
nsys profile ... &
curl http://localhost:8000/...  # Server not ready!
```

**Right:**
```bash
nsys profile ... &
# Wait for health endpoint
for i in $(seq 1 120); do
    curl -s http://localhost:8000/health && break
    sleep 5
done
# Then send requests
```

---

## Profile Quality Indicators

### Good Profile (15-25 MB)

```
$ ls -lh profile.nsys-rep
-rw-r--r-- 1 user user 19M Jan  8 15:30 profile.nsys-rep

$ nsys stats profile.nsys-rep | head
 Time (%)  Total Time (ns)  Instances   Avg (ns)    Med (ns)   Min (ns)   Max (ns)
 --------  ---------------  ---------  ----------  ----------  ---------  ---------
   45.2%    1,234,567,890     1,234    1,000,459   1,000,123    500,000   5,000,000
   ...
```

**Contains:**
- CUDA kernel traces
- cuBLAS/cuDNN operations
- Memory allocations
- CPU thread activity

### Bad Profile (< 1 MB)

```
$ ls -lh profile.nsys-rep
-rw-r--r-- 1 user user 89K Jan  8 15:30 profile.nsys-rep

$ nsys stats profile.nsys-rep
No CUDA data found
```

**Likely Causes:**
- Server wasn't started under nsys
- Server crashed before any inference
- Wrong trace options

---

## Viewing Profiles

### Nsight Systems GUI (Recommended)

```bash
nsys-ui profile.nsys-rep
```

**Key Views:**
1. **Timeline View**: See kernel execution patterns
2. **Statistics View**: Aggregate kernel timing
3. **GPU Metrics**: Utilization over time

### Command Line

```bash
# Summary statistics
nsys stats profile.nsys-rep

# Export to JSON
nsys export --type=json -o profile.json profile.nsys-rep

# GPU kernel summary
nsys stats --report gputrace profile.nsys-rep
```

---

## What to Look for in Analysis

### 1. Model Loading Phase

- Large memory allocations
- Weight loading from disk
- Initial CUDA context setup

### 2. Inference Phase

- Attention kernel patterns
- GEMM operations (matrix multiplications)
- Memory bandwidth utilization
- KV cache management

### 3. Common Bottlenecks

| Symptom | Likely Cause |
|---------|--------------|
| Gaps between kernels | CPU bottleneck |
| Low SM utilization | Memory bound |
| High memory traffic | KV cache thrashing |
| Kernel serialization | Missing parallelism |

---

*Best practices based on profiling vLLM, Triton, and TGI on NVIDIA A10 GPUs*
