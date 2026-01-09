#!/usr/bin/env bash
# Run inference benchmarks
# Usage: ./run_benchmark.sh <backend|all> [--full]
#   backend: vllm, triton, tgi, all
#   --full: Run full benchmark matrix

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="${PROJECT_ROOT}/results/benchmarks"

BACKEND="${1:-all}"
FULL_MODE="${2:-}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "${RESULTS_DIR}"

echo "======================================================="
echo "  LLM Inference Benchmark"
echo "======================================================="
echo ""
echo "Configuration:"
echo "  Backend:     ${BACKEND}"
echo "  Mode:        ${FULL_MODE:-quick}"
echo "  Output:      ${RESULTS_DIR}"
echo ""

# Set benchmark parameters
if [ "$FULL_MODE" == "--full" ]; then
    ITERATIONS=100
    CONCURRENCY_LEVELS="1 8 32"
    MAX_TOKENS_VALUES="128 512"
else
    ITERATIONS=50
    CONCURRENCY_LEVELS="1 8"
    MAX_TOKENS_VALUES="128"
fi

run_benchmark() {
    local backend=$1

    echo "======================================================="
    echo "  Benchmarking: ${backend}"
    echo "======================================================="

    # Check if backend is running
    local deployment_name="${backend}-server"
    if ! kubectl get deployment ${deployment_name} -n bench &> /dev/null; then
        echo "WARNING: ${backend} deployment not found. Skipping."
        return
    fi

    # Get service endpoint
    local service_name="${backend}-service"
    local cluster_ip=$(kubectl get svc ${service_name} -n bench -o jsonpath='{.spec.clusterIP}' 2>/dev/null)

    if [ -z "$cluster_ip" ]; then
        echo "WARNING: ${backend} service not found. Skipping."
        return
    fi

    echo "Service: ${service_name} (${cluster_ip})"
    echo ""

    # Port forward in background
    kubectl port-forward svc/${service_name} 8000:8000 -n bench &
    PF_PID=$!
    sleep 3

    # Run benchmarks
    for concurrency in $CONCURRENCY_LEVELS; do
        for max_tokens in $MAX_TOKENS_VALUES; do
            echo ""
            echo "--- ${backend} | concurrency=${concurrency} | tokens=${max_tokens} ---"

            OUTPUT_FILE="${RESULTS_DIR}/${backend}_c${concurrency}_t${max_tokens}_${TIMESTAMP}.json"

            python3 "${PROJECT_ROOT}/bench-client/inference_client.py" \
                --backend "${backend}" \
                --iterations "${ITERATIONS}" \
                --warmup 10 \
                --max-tokens "${max_tokens}" \
                --concurrency "${concurrency}" \
                --output-file "${OUTPUT_FILE}" || true

            if [ -f "${OUTPUT_FILE}" ]; then
                echo "  Results: ${OUTPUT_FILE}"
            fi
        done
    done

    # Stop port forward
    kill $PF_PID 2>/dev/null || true
}

# Run benchmarks
case $BACKEND in
    vllm|triton|tgi)
        run_benchmark "$BACKEND"
        ;;
    all)
        for b in vllm triton tgi; do
            run_benchmark "$b"
        done
        ;;
    *)
        echo "ERROR: Unknown backend: $BACKEND"
        echo "Supported: vllm, triton, tgi, all"
        exit 1
        ;;
esac

echo ""
echo "======================================================="
echo "  Benchmark Complete"
echo "======================================================="
echo ""
echo "Results saved to: ${RESULTS_DIR}"
echo ""
echo "Generate plots:"
echo "  python3 bench-client/visualize.py --input-dir ${RESULTS_DIR}"
echo ""
