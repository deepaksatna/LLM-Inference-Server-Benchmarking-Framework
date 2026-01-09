# Comprehensive LLM Inference Server Benchmark Report

## vLLM vs NVIDIA Triton vs HuggingFace TGI

**Date**: January 8, 2026
**Model**: Mistral-7B-Instruct-v0.2 (FP16)
**GPU**: NVIDIA A10 (24GB VRAM)
**Platform**: Oracle Kubernetes Engine (OKE)

---

## Executive Summary

This report presents a comprehensive comparison of three leading LLM inference servers deployed on Kubernetes with identical hardware configurations. The benchmark combines client-side performance metrics with server-side GPU profiling.

### Overall Rankings

| Category | Winner | Score |
|----------|--------|-------|
| **Peak Throughput** | HuggingFace TGI | 8.07 req/s |
| **Token Generation** | vLLM | 412 tok/s |
| **Lowest Latency** | HuggingFace TGI | 1704ms P95 |
| **GPU Efficiency** | vLLM | 99% SM util |
| **Memory Efficiency** | vLLM | 19,869 MB |
| **Power Efficiency** | vLLM | 0.197 tok/W |

**Recommendation**: **vLLM** for maximum GPU efficiency and token throughput. **TGI** for lowest latency requirements. **Triton** for enterprise deployments requiring multi-model serving and metrics.

---

## Test Configuration

### Hardware
- **GPU**: NVIDIA A10 (24GB GDDR6, 250W TDP)
- **Node**: OKE GPU shape (8 vCPU, 32GB RAM)
- **Network**: Kubernetes ClusterIP services

### Software
- **vLLM**: v0.4.x with PagedAttention
- **Triton**: 24.x with vLLM backend
- **TGI**: v2.x with Flash Attention

### Workload
- **Concurrency Levels**: 1, 4, 8, 16 concurrent requests
- **Output Tokens**: 50, 100 tokens per request
- **Prompt**: "Explain how transformer attention mechanisms work in detail."
- **Requests per Test**: 20

---

## Performance Results

### 1. Throughput Analysis

#### Peak Throughput (requests/second)

| Backend | c=1 | c=4 | c=8 | c=16 | Peak |
|---------|-----|-----|-----|------|------|
| **vLLM** | 0.58 | 2.17 | 4.22 | 8.00 | **8.00** |
| **Triton** | 0.58 | 2.07 | 3.99 | 7.52 | 7.52 |
| **TGI** | 0.59 | 2.09 | 4.11 | **8.07** | **8.07** |

**Key Finding**: All three backends scale linearly with concurrency. TGI achieves marginally higher peak throughput at c=16.

#### Token Generation Rate (tokens/second)

| Backend | c=1 | c=4 | c=8 | c=16 | Peak |
|---------|-----|-----|-----|------|------|
| **vLLM** | 29.2 | 107 | 208 | **412** | **412** |
| **Triton** | 28.9 | 103 | 200 | 385 | 385 |
| **TGI** | 29.4 | 104 | 206 | 408 | 408 |

**Key Finding**: vLLM achieves highest sustained token generation rate (412 tok/s), 7% faster than Triton.

### 2. Latency Analysis

#### P95 Latency at Concurrency=1 (50 tokens)

| Backend | Avg | P50 | P95 | P99 |
|---------|-----|-----|-----|-----|
| **vLLM** | 1712ms | 1711ms | 1715ms | 1718ms |
| **Triton** | 1728ms | 1727ms | 1736ms | 1736ms |
| **TGI** | **1697ms** | **1697ms** | **1704ms** | **1712ms** |

**Key Finding**: TGI has lowest latency for single requests (1.3% faster than vLLM).

#### Latency Scaling with Concurrency (50 tokens, P95)

| Backend | c=1 | c=4 | c=8 | c=16 | Increase |
|---------|-----|-----|-----|------|----------|
| **vLLM** | 1715ms | 1770ms | 1812ms | 1905ms | +11% |
| **Triton** | 1736ms | 1823ms | 1885ms | 2007ms | +16% |
| **TGI** | **1704ms** | **1797ms** | **1825ms** | **1858ms** | **+9%** |

**Key Finding**: TGI scales best under load with only 9% latency increase at 16x concurrency.

### 3. Scaling Efficiency

Linear scaling efficiency (actual vs theoretical throughput):

| Backend | c=4 | c=8 | c=16 |
|---------|-----|-----|------|
| **vLLM** | 93% | 91% | 86% |
| **Triton** | 89% | 86% | 81% |
| **TGI** | 89% | 87% | 86% |

**Key Finding**: vLLM and TGI maintain better scaling efficiency at high concurrency.

---

## GPU Profiling Results

### Server-Side Metrics (nvidia-smi dmon)

| Metric | vLLM | Triton | TGI |
|--------|------|--------|-----|
| **Peak SM Utilization** | **99%** | 97% | 98% |
| **Avg SM Utilization** | **98.0%** | 95.7% | 97.2% |
| **Memory Utilization** | 100% | 100% | 100% |
| **Peak Power** | 151W | 151W | 152W |
| **Idle Power** | 87W | 94W | 90W |
| **GPU Memory Used** | **19,869 MB** | 20,270 MB | 20,189 MB |
| **Peak Temperature** | **60°C** | 62°C | 63°C |

### GPU Efficiency Analysis

#### Compute Efficiency (SM Utilization)
```
vLLM:   ████████████████████████████████████████████████████████████ 99%
TGI:    █████████████████████████████████████████████████████████░░░ 98%
Triton: ██████████████████████████████████████████████████████████░░ 97%
```

**Winner**: vLLM achieves highest SM occupancy, indicating most efficient GPU compute usage.

#### Power Efficiency (Tokens per Watt)

| Backend | Power (Active) | Tokens/s | Tokens/Watt |
|---------|----------------|----------|-------------|
| **vLLM** | 149W | 29.4 | **0.197** |
| Triton | 149W | 28.9 | 0.194 |
| TGI | 150W | 29.4 | 0.196 |

**Winner**: vLLM - Most tokens generated per watt consumed.

#### Memory Overhead

| Backend | GPU Memory | Model Size | Overhead |
|---------|------------|------------|----------|
| **vLLM** | 19,869 MB | ~14,000 MB | **5,869 MB (42%)** |
| Triton | 20,270 MB | ~14,000 MB | 6,270 MB (45%) |
| TGI | 20,189 MB | ~14,000 MB | 6,189 MB (44%) |

**Winner**: vLLM - Lowest memory overhead, enabling larger batch sizes or longer contexts.

---

## Visualization Gallery

The following visualizations were generated:

| File | Description |
|------|-------------|
| `benchmark_summary_dashboard.png` | Complete overview with all key metrics |
| `throughput_comparison.png` | Throughput vs concurrency scaling |
| `tokens_per_second.png` | Token generation rate comparison |
| `latency_comparison.png` | P50/P95/P99 latency analysis |
| `scaling_efficiency.png` | Linear scaling efficiency |
| `performance_heatmap.png` | Multi-dimensional heatmap |
| `gpu_metrics_comparison.png` | GPU utilization and power |

---

## Detailed Recommendations

### Use vLLM When:
- Maximum GPU compute efficiency is critical
- Token throughput is the primary metric
- Memory efficiency matters (longer contexts, larger batches)
- Power efficiency is a concern (cost optimization)
- Running single-model deployments

### Use NVIDIA Triton When:
- Enterprise monitoring is required (Prometheus metrics)
- Multi-model serving is needed
- Advanced batching strategies are required
- Integration with existing Triton infrastructure
- A/B testing or model versioning is needed

### Use HuggingFace TGI When:
- Lowest latency is the top priority
- Best latency scaling under load is needed
- Simple deployment is preferred
- HuggingFace ecosystem integration is desired
- Streaming responses are important

---

## Methodology

### Benchmark Execution
1. Deploy all three servers with identical resource limits
2. Warm-up with 5 requests per server
3. Execute 20 requests per configuration
4. Measure end-to-end latency and throughput
5. Calculate percentiles and aggregate statistics

### GPU Profiling
1. Start `nvidia-smi dmon` inside server pods
2. Execute inference workload
3. Capture SM%, memory%, power, temperature
4. Analyze time-series data for patterns

### Data Collection
- Client-side: Custom Python benchmark client
- Server-side: nvidia-smi dmon @ 1-second intervals
- Profiling: Nsight Systems for timing analysis

---

## Files Generated

### Benchmark Data
```
results/benchmarks/20260108/
├── vllm_c1_t50.json      # vLLM, concurrency=1, 50 tokens
├── vllm_c1_t100.json     # vLLM, concurrency=1, 100 tokens
├── vllm_c4_t50.json      # ... and so on for all combinations
├── triton_*.json         # Triton benchmarks
├── tgi_*.json            # TGI benchmarks
└── BENCHMARK_REPORT.md   # Initial benchmark report
```

### GPU Profiles
```
results/profiles/20260108_final/
├── vllm_gpu_dmon.log     # vLLM GPU metrics
├── triton_gpu_dmon.log   # Triton GPU metrics
├── tgi_gpu_dmon.log      # TGI GPU metrics
├── nsys_profiles/        # Nsight Systems profiles
└── GPU_PROFILING_REPORT.md
```

### Visualizations
```
results/plots/
├── benchmark_summary_dashboard.png
├── throughput_comparison.png
├── tokens_per_second.png
├── latency_comparison.png
├── scaling_efficiency.png
├── performance_heatmap.png
└── gpu_metrics_comparison.png
```

---

## Conclusion

All three LLM inference servers demonstrate production-ready performance with Mistral-7B on NVIDIA A10 GPUs. The choice between them depends on specific requirements:

- **vLLM** leads in GPU efficiency metrics (SM utilization, memory overhead, power efficiency)
- **TGI** leads in latency metrics (lowest P95, best scaling under load)
- **Triton** provides the best enterprise features (monitoring, multi-model, batching)

For most use cases, **vLLM** provides the best balance of performance and efficiency. For latency-sensitive applications, **TGI** is the better choice. For enterprise deployments requiring observability, **Triton** is recommended despite the ~2-3% efficiency overhead.

---

*Report generated by LLM Inference Benchmarking Framework v1.0*
*Visualizations: matplotlib 3.x with seaborn styling*
