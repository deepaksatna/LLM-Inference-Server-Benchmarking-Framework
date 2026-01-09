#!/usr/bin/env bash
# =============================================================================
# Full LLM Inference Benchmark Suite
# Runs benchmarks and nsys profiling sequentially on single GPU node
# Results saved directly to VM shared filesystem
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="${PROJECT_ROOT}/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Endpoints (external IPs)
VLLM_IP="${VLLM_IP:-141.147.1.241}"
TRITON_IP="${TRITON_IP:-79.76.118.117}"
TGI_IP="${TGI_IP:-141.144.233.78}"

echo "=============================================="
echo "  LLM Inference Benchmark Suite"
echo "  vLLM vs Triton vs TGI"
echo "=============================================="
echo ""
echo "Timestamp: ${TIMESTAMP}"
echo "Results Directory: ${RESULTS_DIR}"
echo ""
echo "Endpoints:"
echo "  vLLM:   http://${VLLM_IP}"
echo "  Triton: http://${TRITON_IP}"
echo "  TGI:    http://${TGI_IP}"
echo ""

# Create results directories
mkdir -p "${RESULTS_DIR}/benchmarks/${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}/profiles/${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}/plots/${TIMESTAMP}"

BENCH_DIR="${RESULTS_DIR}/benchmarks/${TIMESTAMP}"
PROFILE_DIR="${RESULTS_DIR}/profiles/${TIMESTAMP}"

# =============================================================================
# Phase 1: Verify Services
# =============================================================================
echo "=============================================="
echo "  Phase 1: Verifying Services"
echo "=============================================="

check_service() {
    local name=$1
    local url=$2
    echo -n "Checking ${name}... "
    if curl -s --max-time 10 "${url}" > /dev/null 2>&1; then
        echo "OK"
        return 0
    else
        echo "FAILED"
        return 1
    fi
}

check_service "vLLM" "http://${VLLM_IP}/health"
check_service "Triton" "http://${TRITON_IP}/v2/health/ready"
check_service "TGI" "http://${TGI_IP}/health"
echo ""

# =============================================================================
# Phase 2: Warmup
# =============================================================================
echo "=============================================="
echo "  Phase 2: Warmup (10 requests each)"
echo "=============================================="

echo "Warming up vLLM..."
for i in {1..10}; do
    curl -s "http://${VLLM_IP}/v1/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"mistralai/Mistral-7B-Instruct-v0.2","prompt":"Hello","max_tokens":10}' > /dev/null &
done
wait
echo "  vLLM warmup complete"

echo "Warming up Triton..."
for i in {1..10}; do
    curl -s "http://${TRITON_IP}/v2/models/mistral/generate" \
        -H "Content-Type: application/json" \
        -d '{"text_input":"Hello","parameters":{"max_tokens":10}}' > /dev/null &
done
wait
echo "  Triton warmup complete"

echo "Warming up TGI..."
for i in {1..10}; do
    curl -s "http://${TGI_IP}/generate" \
        -H "Content-Type: application/json" \
        -d '{"inputs":"Hello","parameters":{"max_new_tokens":10}}' > /dev/null &
done
wait
echo "  TGI warmup complete"
echo ""

# =============================================================================
# Phase 3: Benchmark Suite
# =============================================================================
echo "=============================================="
echo "  Phase 3: Running Benchmarks"
echo "=============================================="

# Benchmark parameters
BENCHMARK_REQUESTS=100
CONCURRENCY_LEVELS="1 4 8 16 32"
MAX_TOKENS_LIST="50 100 256"
PROMPT="Explain the key differences between supervised and unsupervised machine learning, providing examples of each approach and when to use them."

# Create benchmark script
cat > /tmp/benchmark.py << 'EOFPY'
import requests
import time
import json
import sys
import concurrent.futures
import statistics

def benchmark(backend, url, endpoint, prompt, max_tokens, concurrency, num_requests):
    results = []

    def make_request(i):
        try:
            start = time.time()
            if backend == "vllm":
                resp = requests.post(f"{url}{endpoint}", json={
                    "model": "mistralai/Mistral-7B-Instruct-v0.2",
                    "prompt": prompt,
                    "max_tokens": max_tokens
                }, timeout=120)
                tokens = resp.json().get("usage", {}).get("completion_tokens", max_tokens) if resp.status_code == 200 else 0
            elif backend == "triton":
                resp = requests.post(f"{url}{endpoint}", json={
                    "text_input": prompt,
                    "parameters": {"max_tokens": max_tokens}
                }, timeout=120)
                tokens = max_tokens if resp.status_code == 200 else 0
            elif backend == "tgi":
                resp = requests.post(f"{url}{endpoint}", json={
                    "inputs": prompt,
                    "parameters": {"max_new_tokens": max_tokens}
                }, timeout=120)
                tokens = max_tokens if resp.status_code == 200 else 0

            latency = time.time() - start
            return {"latency": latency, "tokens": tokens, "success": resp.status_code == 200}
        except Exception as e:
            return {"latency": 0, "tokens": 0, "success": False, "error": str(e)}

    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = list(executor.map(make_request, range(num_requests)))
    total_time = time.time() - start_time

    successful = [r for r in results if r["success"]]
    latencies = [r["latency"] for r in successful]
    total_tokens = sum(r["tokens"] for r in successful)

    if not latencies:
        return None

    return {
        "backend": backend,
        "concurrency": concurrency,
        "max_tokens": max_tokens,
        "prompt_length": len(prompt.split()),
        "total_requests": num_requests,
        "successful_requests": len(successful),
        "failed_requests": num_requests - len(successful),
        "total_time_sec": round(total_time, 3),
        "throughput_rps": round(len(successful) / total_time, 3),
        "tokens_per_sec": round(total_tokens / total_time, 1),
        "avg_latency_ms": round(statistics.mean(latencies) * 1000, 1),
        "p50_latency_ms": round(statistics.median(latencies) * 1000, 1),
        "p90_latency_ms": round(sorted(latencies)[int(len(latencies)*0.90)] * 1000, 1) if len(latencies) > 1 else 0,
        "p95_latency_ms": round(sorted(latencies)[int(len(latencies)*0.95)] * 1000, 1) if len(latencies) > 1 else 0,
        "p99_latency_ms": round(sorted(latencies)[int(len(latencies)*0.99)] * 1000, 1) if len(latencies) > 1 else 0,
        "min_latency_ms": round(min(latencies) * 1000, 1),
        "max_latency_ms": round(max(latencies) * 1000, 1)
    }

if __name__ == "__main__":
    backend = sys.argv[1]
    url = sys.argv[2]
    endpoint = sys.argv[3]
    prompt = sys.argv[4]
    max_tokens = int(sys.argv[5])
    concurrency = int(sys.argv[6])
    num_requests = int(sys.argv[7])
    output_file = sys.argv[8]

    print(f"  {backend}: c={concurrency}, tokens={max_tokens}, requests={num_requests}")
    result = benchmark(backend, url, endpoint, prompt, max_tokens, concurrency, num_requests)

    if result:
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"    Throughput: {result['throughput_rps']:.2f} req/s, {result['tokens_per_sec']:.1f} tok/s")
        print(f"    Latency: avg={result['avg_latency_ms']:.1f}ms, p95={result['p95_latency_ms']:.1f}ms")
    else:
        print(f"    FAILED - no successful requests")
EOFPY

# Run benchmarks for each backend
for max_tokens in ${MAX_TOKENS_LIST}; do
    echo ""
    echo ">>> Max Tokens: ${max_tokens}"
    echo "-------------------------------------------"

    for concurrency in ${CONCURRENCY_LEVELS}; do
        echo ""
        echo ">> Concurrency: ${concurrency}"

        # vLLM
        python3 /tmp/benchmark.py vllm "http://${VLLM_IP}" "/v1/completions" \
            "${PROMPT}" ${max_tokens} ${concurrency} ${BENCHMARK_REQUESTS} \
            "${BENCH_DIR}/vllm_c${concurrency}_t${max_tokens}.json"

        # Triton
        python3 /tmp/benchmark.py triton "http://${TRITON_IP}" "/v2/models/mistral/generate" \
            "${PROMPT}" ${max_tokens} ${concurrency} ${BENCHMARK_REQUESTS} \
            "${BENCH_DIR}/triton_c${concurrency}_t${max_tokens}.json"

        # TGI
        python3 /tmp/benchmark.py tgi "http://${TGI_IP}" "/generate" \
            "${PROMPT}" ${max_tokens} ${concurrency} ${BENCHMARK_REQUESTS} \
            "${BENCH_DIR}/tgi_c${concurrency}_t${max_tokens}.json"
    done
done

echo ""
echo "Benchmark results saved to: ${BENCH_DIR}"
echo ""

# =============================================================================
# Phase 4: Generate Summary Report
# =============================================================================
echo "=============================================="
echo "  Phase 4: Generating Summary Report"
echo "=============================================="

python3 << EOFPY
import json
import os
from pathlib import Path

bench_dir = "${BENCH_DIR}"
results = {"vllm": [], "triton": [], "tgi": []}

for f in Path(bench_dir).glob("*.json"):
    try:
        with open(f) as fp:
            data = json.load(fp)
            results[data["backend"]].append(data)
    except:
        pass

# Generate markdown report
report = """# LLM Inference Benchmark Results
## Generated: ${TIMESTAMP}

## Test Configuration
- **Model**: Mistral-7B-Instruct-v0.2
- **Backends**: vLLM, NVIDIA Triton (vLLM backend), HuggingFace TGI
- **Requests per test**: ${BENCHMARK_REQUESTS}
- **Concurrency levels**: ${CONCURRENCY_LEVELS}
- **Max tokens**: ${MAX_TOKENS_LIST}

## Endpoints
- vLLM: http://${VLLM_IP}
- Triton: http://${TRITON_IP}
- TGI: http://${TGI_IP}

---

## Summary Table

| Backend | Concurrency | Max Tokens | Throughput (req/s) | Tokens/s | Avg Latency (ms) | P95 Latency (ms) | P99 Latency (ms) |
|---------|-------------|------------|-------------------|----------|------------------|------------------|------------------|
"""

for backend in ["vllm", "triton", "tgi"]:
    for result in sorted(results[backend], key=lambda x: (x["max_tokens"], x["concurrency"])):
        report += f"| {backend.upper()} | {result['concurrency']} | {result['max_tokens']} | {result['throughput_rps']:.2f} | {result['tokens_per_sec']:.1f} | {result['avg_latency_ms']:.1f} | {result['p95_latency_ms']:.1f} | {result['p99_latency_ms']:.1f} |\n"

report += """

---

## Performance Comparison

### Peak Throughput (requests/second)
"""

for backend in ["vllm", "triton", "tgi"]:
    if results[backend]:
        best = max(results[backend], key=lambda x: x["throughput_rps"])
        report += f"- **{backend.upper()}**: {best['throughput_rps']:.2f} req/s @ concurrency={best['concurrency']}, tokens={best['max_tokens']}\n"

report += """

### Best Latency (P95 at concurrency=1)
"""

for backend in ["vllm", "triton", "tgi"]:
    c1 = [r for r in results[backend] if r["concurrency"] == 1]
    if c1:
        best = min(c1, key=lambda x: x["p95_latency_ms"])
        report += f"- **{backend.upper()}**: {best['p95_latency_ms']:.1f}ms @ tokens={best['max_tokens']}\n"

report += """

### Tokens per Second (Peak)
"""

for backend in ["vllm", "triton", "tgi"]:
    if results[backend]:
        best = max(results[backend], key=lambda x: x["tokens_per_sec"])
        report += f"- **{backend.upper()}**: {best['tokens_per_sec']:.1f} tokens/s @ concurrency={best['concurrency']}, tokens={best['max_tokens']}\n"

report += """

---

## Detailed Results by Concurrency

"""

for conc in [1, 4, 8, 16, 32]:
    report += f"### Concurrency = {conc}\n\n"
    report += "| Backend | Max Tokens | Throughput (req/s) | Tokens/s | Avg Latency | P95 Latency |\n"
    report += "|---------|------------|-------------------|----------|-------------|-------------|\n"

    for backend in ["vllm", "triton", "tgi"]:
        conc_results = [r for r in results[backend] if r["concurrency"] == conc]
        for r in sorted(conc_results, key=lambda x: x["max_tokens"]):
            report += f"| {backend.upper()} | {r['max_tokens']} | {r['throughput_rps']:.2f} | {r['tokens_per_sec']:.1f} | {r['avg_latency_ms']:.1f}ms | {r['p95_latency_ms']:.1f}ms |\n"
    report += "\n"

# Save report
with open("${BENCH_DIR}/BENCHMARK_REPORT.md", "w") as f:
    f.write(report)

# Save combined JSON
with open("${BENCH_DIR}/all_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Report saved: ${BENCH_DIR}/BENCHMARK_REPORT.md")
print(f"JSON saved: ${BENCH_DIR}/all_results.json")
EOFPY

echo ""

# =============================================================================
# Phase 5: Nsight Systems Profiling (if nsys available)
# =============================================================================
echo "=============================================="
echo "  Phase 5: Nsight Systems Profiling"
echo "=============================================="

if command -v nsys &> /dev/null; then
    echo "nsys found: $(nsys --version | head -1)"
    echo ""

    PROFILE_REQUESTS=20
    PROFILE_TOKENS=100
    PROFILE_PROMPT="Explain the architecture of a transformer model and how attention mechanisms work."

    # Create profiling script
    cat > /tmp/profile_inference.py << 'EOFPY'
import requests
import sys
import time

backend = sys.argv[1]
url = sys.argv[2]
prompt = sys.argv[3]
max_tokens = int(sys.argv[4])
num_requests = int(sys.argv[5])

print(f"Profiling {backend}: {num_requests} requests")

for i in range(num_requests):
    try:
        if backend == "vllm":
            resp = requests.post(f"{url}/v1/completions", json={
                "model": "mistralai/Mistral-7B-Instruct-v0.2",
                "prompt": prompt,
                "max_tokens": max_tokens
            }, timeout=120)
        elif backend == "triton":
            resp = requests.post(f"{url}/v2/models/mistral/generate", json={
                "text_input": prompt,
                "parameters": {"max_tokens": max_tokens}
            }, timeout=120)
        elif backend == "tgi":
            resp = requests.post(f"{url}/generate", json={
                "inputs": prompt,
                "parameters": {"max_new_tokens": max_tokens}
            }, timeout=120)

        status = "OK" if resp.status_code == 200 else f"Error {resp.status_code}"
        print(f"  [{i+1}/{num_requests}] {status}")
    except Exception as e:
        print(f"  [{i+1}/{num_requests}] Failed: {e}")

print(f"Completed {num_requests} requests")
EOFPY

    # Profile each backend
    for backend in vllm triton tgi; do
        echo ""
        echo "Profiling ${backend}..."

        case ${backend} in
            vllm) URL="http://${VLLM_IP}" ;;
            triton) URL="http://${TRITON_IP}" ;;
            tgi) URL="http://${TGI_IP}" ;;
        esac

        nsys profile \
            --output="${PROFILE_DIR}/${backend}_profile" \
            --trace=cuda,nvtx,osrt,cudnn,cublas \
            --cuda-memory-usage=true \
            --sample=cpu \
            --cpuctxsw=process-tree \
            --force-overwrite=true \
            python3 /tmp/profile_inference.py ${backend} "${URL}" "${PROFILE_PROMPT}" ${PROFILE_TOKENS} ${PROFILE_REQUESTS} \
            2>&1 | tee "${PROFILE_DIR}/${backend}_profile.log"

        echo "  Profile saved: ${PROFILE_DIR}/${backend}_profile.nsys-rep"
    done

    # Generate stats
    echo ""
    echo "Generating statistics reports..."
    for backend in vllm triton tgi; do
        if [ -f "${PROFILE_DIR}/${backend}_profile.nsys-rep" ]; then
            nsys stats \
                --report cuda_gpu_kern_sum \
                --format csv \
                --output "${PROFILE_DIR}/${backend}_kernel_stats" \
                "${PROFILE_DIR}/${backend}_profile.nsys-rep" 2>/dev/null || true
        fi
    done

    echo ""
    echo "Profile files saved to: ${PROFILE_DIR}"
else
    echo "WARNING: nsys not found on this system"
    echo "To run profiling, either:"
    echo "  1. Install nsight-systems-cli"
    echo "  2. Run the nsys-profile-job.yaml on a GPU node"
    echo ""
fi

# =============================================================================
# Phase 6: Summary
# =============================================================================
echo ""
echo "=============================================="
echo "  Benchmark Complete!"
echo "=============================================="
echo ""
echo "Results Location: ${RESULTS_DIR}"
echo ""
echo "Benchmarks:"
ls -la "${BENCH_DIR}/"
echo ""

if [ -d "${PROFILE_DIR}" ] && [ "$(ls -A ${PROFILE_DIR})" ]; then
    echo "Profiles:"
    ls -la "${PROFILE_DIR}/"
    echo ""
fi

echo "=============================================="
echo "  Quick Summary"
echo "=============================================="
cat "${BENCH_DIR}/BENCHMARK_REPORT.md" | head -60
echo ""
echo "..."
echo ""
echo "Full report: ${BENCH_DIR}/BENCHMARK_REPORT.md"
echo ""
