# GPU-Level Profiling Report: LLM Inference Servers

## Generated: January 8, 2026

---

## Executive Summary

This report presents GPU-level profiling results captured **inside the inference server pods** during active inference workloads. Unlike client-side profiling, these metrics capture actual GPU utilization, power consumption, and memory bandwidth during token generation.

### Key Findings

| Metric | vLLM | Triton | TGI |
|--------|------|--------|-----|
| **Peak SM Utilization** | **99%** | 97% | 98% |
| **Memory Utilization** | 100% | 100% | 100% |
| **Peak Power Draw** | 151W | 151W | **152W** |
| **Idle Power** | 87W | 94W | 90W |
| **GPU Memory Used** | 19,869 MB | 20,270 MB | 20,189 MB |
| **Temperature Rise** | 54°C → 60°C | 57°C → 62°C | 58°C → 63°C |

**Verdict**: vLLM achieves the highest SM utilization (99%), indicating more efficient GPU compute usage. All three backends saturate GPU memory bandwidth (100% memory utilization) during inference.

---

## Test Configuration

- **Model**: Mistral-7B-Instruct-v0.2 (FP16)
- **GPU**: NVIDIA A10 (24GB VRAM, 250W TDP)
- **Workload**: 5 sequential requests, 100 tokens each
- **Monitoring**: `nvidia-smi dmon` @ 1-second intervals
- **Prompt**: "Explain how transformer attention mechanisms work in detail."

---

## Detailed GPU Metrics

### 1. vLLM Server

```
Idle State:
  Power: 87W | SM: 0% | Memory: 0% | Temp: 54°C

Active Inference:
  Power: 148-151W | SM: 97-99% | Memory: 100% | Temp: 60°C

Key Observations:
  - Fastest ramp-up to peak utilization
  - Highest sustained SM utilization (98-99%)
  - Consistent power draw during generation
  - Memory utilization: 19,869 MB / 23,028 MB (86.3%)
```

**Raw Data (during inference):**
```
# gpu  pwr  temp   sm   mem   mclk   pclk
   0   149   56   98%  100%  6251   1350
   0   149   57   98%  100%  6251   1335
   0   148   57   97%  100%  6251   1335
   0   151   57   98%  100%  6251   1335
   0   150   58   98%  100%  6251   1335
   0   149   58   98%  100%  6251   1320
   0   148   58   98%  100%  6251   1320
   0   150   59   98%  100%  6251   1305
   0   151   59   98%  100%  6251   1290
   0   149   59   98%  100%  6251   1290
   0   147   59   98%  100%  6251   1305
   0   150   59   98%  100%  6251   1290
   0   151   60   98%  100%  6251   1290
   0   149   60   98%  100%  6251   1290
   0   148   60   98%  100%  6251   1305
   0   148   60   99%  100%  6251   1290
```

---

### 2. NVIDIA Triton Server (vLLM Backend)

```
Idle State:
  Power: 94W | SM: 0% | Memory: 0% | Temp: 57°C

Active Inference:
  Power: 147-151W | SM: 95-97% | Memory: 100% | Temp: 62°C

Key Observations:
  - Slightly lower SM utilization than native vLLM
  - Triton framework adds ~1-2% overhead
  - Stable power consumption during generation
  - Memory utilization: 20,270 MB / 23,028 MB (88.0%)
```

**Raw Data (during inference):**
```
# gpu  pwr  temp   sm   mem   mclk   pclk
   0    99   57   95%  100%  6251   1470
   0   149   58   96%  100%  6251   1290
   0   149   59   96%  100%  6251   1275
   0   149   59   96%  100%  6251   1275
   0   147   59   96%  100%  6251   1290
   0   148   60   96%  100%  6251   1260
   0   150   60   96%  100%  6251   1275
   0   151   60   96%  100%  6251   1305
   0   150   60   96%  100%  6251   1275
   0   150   60   96%  100%  6251   1275
   0   149   61   96%  100%  6251   1260
   0   149   61   96%  100%  6251   1260
   0   149   61   96%  100%  6251   1275
   0   149   61   90%   98%  6251   1305
   0   150   61   97%  100%  6251   1260
```

---

### 3. HuggingFace TGI Server

```
Idle State:
  Power: 90W | SM: 0% | Memory: 0% | Temp: 58°C

Active Inference:
  Power: 148-152W | SM: 97-98% | Memory: 100% | Temp: 63°C

Key Observations:
  - Highest peak power draw (152W)
  - SM utilization between vLLM and Triton
  - Slightly higher temperature during inference
  - Memory utilization: 20,189 MB / 23,028 MB (87.7%)
```

**Raw Data (during inference):**
```
# gpu  pwr  temp   sm   mem   mclk   pclk
   0   151   60   97%  100%  6251   1320
   0   152   60   97%  100%  6251   1320
   0   149   61   98%  100%  6251   1290
   0   151   61   98%  100%  6251   1260
   0   148   61   97%  100%  6251   1305
   0   151   61   97%  100%  6251   1305
   0   151   62   97%  100%  6251   1275
   0   150   62   97%  100%  6251   1275
   0   149   62   97%  100%  6251   1275
   0   149   62   97%  100%  6251   1275
   0   151   63   97%  100%  6251   1290
   0   148   63   97%  100%  6251   1275
   0   151   63   97%  100%  6251   1260
   0   148   63   97%  100%  6251   1260
   0   151   63   97%  100%  6251   1275
   0   148   63   98%  100%  6251   1575
```

---

## Comparative Analysis

### GPU Compute Efficiency (SM Utilization)

```
vLLM:   ████████████████████████████████████████████████████████████ 99%
TGI:    █████████████████████████████████████████████████████████░░░ 98%
Triton: ██████████████████████████████████████████████████████████░░ 97%
```

**Winner: vLLM** - Achieves highest SM occupancy, indicating most efficient use of GPU compute resources.

### Power Efficiency (Tokens per Watt)

Based on throughput benchmarks (29.4 tok/s @ c=1) and power draw:

| Backend | Power (Active) | Tokens/s | Tokens/Watt |
|---------|----------------|----------|-------------|
| vLLM | 149W | 29.4 | **0.197** |
| Triton | 149W | 28.9 | 0.194 |
| TGI | 150W | 29.4 | 0.196 |

**Winner: vLLM** - Most tokens generated per watt consumed.

### Memory Efficiency

| Backend | GPU Memory | Model Size | Overhead |
|---------|------------|------------|----------|
| vLLM | 19,869 MB | ~14,000 MB | 5,869 MB (42%) |
| Triton | 20,270 MB | ~14,000 MB | 6,270 MB (45%) |
| TGI | 20,189 MB | ~14,000 MB | 6,189 MB (44%) |

**Winner: vLLM** - Lowest memory overhead, enabling larger batch sizes or longer contexts.

### Thermal Performance

| Backend | Idle Temp | Active Temp | Delta |
|---------|-----------|-------------|-------|
| vLLM | 54°C | 60°C | +6°C |
| Triton | 57°C | 62°C | +5°C |
| TGI | 58°C | 63°C | +5°C |

All backends operate well within thermal limits (A10 throttles at 96°C).

---

## GPU Clock Behavior

All backends exhibit dynamic frequency scaling:

- **Memory Clock**: Constant 6251 MHz (maximum for A10)
- **GPU Clock**: Scales between 1260-1695 MHz based on workload
  - Idle: 1695 MHz (low power state)
  - Active: 1260-1350 MHz (thermal/power throttling)

The clock throttling from ~1695 MHz to ~1300 MHz indicates the GPU is hitting power limits (150W sustained), not thermal limits.

---

## Recommendations

### For Maximum GPU Efficiency
**Use vLLM** - Achieves highest SM utilization and best memory efficiency.

### For Enterprise Deployments with Monitoring
**Use Triton** - Slightly lower efficiency but includes:
- Built-in Prometheus metrics (`/metrics` endpoint)
- Multi-model serving capability
- Advanced batching controls

### For Simple High-Performance Deployment
**Use TGI** - Good balance of efficiency and ease of deployment.

---

## Files Generated

### Nsight Systems Profile Files (.nsys-rep)

These can be opened in Nsight Systems GUI for timeline visualization:

| File | Type | Contents | Size |
|------|------|----------|------|
| `nsys_profiles/vllm_inference_profile.nsys-rep` | Client-side | Request timing, OS runtime | 97 KB |
| `nsys_profiles/triton_inference_profile.nsys-rep` | Client-side | Request timing, OS runtime | 98 KB |
| `nsys_profiles/tgi_inference_profile.nsys-rep` | Client-side | Request timing, OS runtime | 97 KB |
| `triton_gpu_profile.nsys-rep` | Server attempt | Triton startup profile | 247 KB |

### GPU Metrics Logs (nvidia-smi dmon)

Real server-side GPU metrics during inference:

| File | Description | Size |
|------|-------------|------|
| `vllm_gpu_dmon.log` | vLLM: SM%, memory%, power, temp | 3.5 KB |
| `triton_gpu_dmon.log` | Triton: SM%, memory%, power, temp | 20.8 KB |
| `tgi_gpu_dmon.log` | TGI: SM%, memory%, power, temp | 3.4 KB |

### How to View Profiles

```bash
# Open in Nsight Systems GUI (recommended for timeline view)
nsys-ui nsys_profiles/vllm_inference_profile.nsys-rep

# Generate summary stats via command line
nsys stats --report osrt_sum nsys_profiles/vllm_inference_profile.nsys-rep

# Export to different formats
nsys export --type=json -o vllm_profile.json nsys_profiles/vllm_inference_profile.nsys-rep
```

### Profile Contents

The nsys profiles contain:
- **OS Runtime Analysis**: Time spent in poll(), recv(), send() etc.
- **Thread Activity**: CPU thread scheduling and context switches
- **Request Timing**: End-to-end latency per request

The nvidia-smi logs contain:
- **SM Utilization**: GPU compute unit usage (%)
- **Memory Utilization**: Memory bandwidth usage (%)
- **Power Draw**: Watts consumed
- **Temperature**: GPU die temperature (Celsius)
- **Clock Frequencies**: Memory and GPU clocks (MHz)

---

## Methodology

### nvidia-smi dmon Parameters
```bash
nvidia-smi dmon -s pucvmet -d 1
```
- `-s pucvmet`: Power, Utilization, Clocks, Violations, Memory, ECC, Temperature
- `-d 1`: 1-second sampling interval

### Profiling Approach
1. Start `nvidia-smi dmon` in background on server pod
2. Execute 5 inference requests (100 tokens each)
3. Capture GPU metrics during active inference
4. Stop monitoring and collect results

This approach captures real server-side GPU activity, unlike client-side profiling which only sees network latency.

---

*Report generated by LLM Inference Benchmarking Framework v1.0*
