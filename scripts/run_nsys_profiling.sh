#!/usr/bin/env bash
# =============================================================================
# Nsight Systems Profiling Script
# Deploys bench-client pod with GPU, runs nsys profiling, copies results back
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="${PROJECT_ROOT}/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROFILE_DIR="${RESULTS_DIR}/profiles/${TIMESTAMP}"
NAMESPACE="bench"

echo "=============================================="
echo "  Nsight Systems Profiling for LLM Inference"
echo "  vLLM vs Triton vs TGI"
echo "=============================================="
echo ""
echo "Timestamp: ${TIMESTAMP}"
echo "Profile Directory: ${PROFILE_DIR}"
echo ""

# Create local results directory
mkdir -p "${PROFILE_DIR}"

# =============================================================================
# Step 1: Deploy bench-client
# =============================================================================
echo "=============================================="
echo "  Step 1: Deploying bench-client pod"
echo "=============================================="

# Check if already deployed
if kubectl get deployment bench-client -n ${NAMESPACE} &> /dev/null; then
    echo "bench-client deployment exists, restarting..."
    kubectl rollout restart deployment/bench-client -n ${NAMESPACE}
else
    echo "Creating bench-client deployment..."
    kubectl apply -f "${PROJECT_ROOT}/k8s/bench-client/nsys-profiler-deployment.yaml"
fi

# Wait for pod to be ready
echo "Waiting for bench-client pod to be ready..."
kubectl wait --for=condition=ready pod -l app=bench-client -n ${NAMESPACE} --timeout=300s

POD_NAME=$(kubectl get pod -l app=bench-client -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}')
echo "Pod ready: ${POD_NAME}"
echo ""

# Verify GPU and nsys in pod
echo "Verifying GPU access in pod..."
kubectl exec ${POD_NAME} -n ${NAMESPACE} -- nvidia-smi --query-gpu=name,memory.total --format=csv

echo ""
echo "Verifying nsys in pod..."
kubectl exec ${POD_NAME} -n ${NAMESPACE} -- nsys --version

echo ""

# =============================================================================
# Step 2: Create profiling script inside pod
# =============================================================================
echo "=============================================="
echo "  Step 2: Setting up profiling environment"
echo "=============================================="

kubectl exec ${POD_NAME} -n ${NAMESPACE} -- bash -c 'cat > /tmp/run_profile.py << "EOFPY"
import requests
import sys
import time

backend = sys.argv[1]
num_requests = int(sys.argv[2])
max_tokens = int(sys.argv[3])

# Internal service URLs
urls = {
    "vllm": "http://vllm-service.bench.svc.cluster.local:8000",
    "triton": "http://triton-service.bench.svc.cluster.local:8000",
    "tgi": "http://tgi-service.bench.svc.cluster.local:8000"
}

prompt = "Explain in detail how transformer neural networks work, including the attention mechanism, positional encoding, and how they differ from recurrent neural networks."

url = urls[backend]
print(f"Profiling {backend} at {url}")
print(f"Requests: {num_requests}, Max tokens: {max_tokens}")
print("")

for i in range(num_requests):
    try:
        start = time.time()
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

        latency = time.time() - start
        status = "OK" if resp.status_code == 200 else f"Error {resp.status_code}"
        print(f"  [{i+1}/{num_requests}] {status} - {latency:.2f}s")
    except Exception as e:
        print(f"  [{i+1}/{num_requests}] Failed: {str(e)[:50]}")

print(f"\nCompleted {num_requests} requests for {backend}")
EOFPY'

echo "Profiling script created in pod"
echo ""

# =============================================================================
# Step 3: Run warmup
# =============================================================================
echo "=============================================="
echo "  Step 3: Warmup (5 requests each backend)"
echo "=============================================="

for backend in vllm triton tgi; do
    echo "Warming up ${backend}..."
    kubectl exec ${POD_NAME} -n ${NAMESPACE} -- python3 /tmp/run_profile.py ${backend} 5 50 2>&1 | tail -2
done
echo ""

# =============================================================================
# Step 4: Run nsys profiling for each backend
# =============================================================================
echo "=============================================="
echo "  Step 4: Running Nsight Systems Profiling"
echo "=============================================="

PROFILE_REQUESTS=20
PROFILE_TOKENS=100

for backend in vllm triton tgi; do
    echo ""
    echo ">>> Profiling ${backend} (${PROFILE_REQUESTS} requests, ${PROFILE_TOKENS} tokens)"
    echo "-------------------------------------------"

    kubectl exec ${POD_NAME} -n ${NAMESPACE} -- bash -c "
        nsys profile \
            --output=/results/${backend}_profile \
            --trace=cuda,nvtx,osrt,cudnn,cublas \
            --cuda-memory-usage=true \
            --sample=cpu \
            --cpuctxsw=process-tree \
            --force-overwrite=true \
            --stats=true \
            python3 /tmp/run_profile.py ${backend} ${PROFILE_REQUESTS} ${PROFILE_TOKENS}
    " 2>&1 | tee "${PROFILE_DIR}/${backend}_nsys.log"

    echo ""
    echo "${backend} profiling complete!"
done

echo ""

# =============================================================================
# Step 5: Generate statistics reports
# =============================================================================
echo "=============================================="
echo "  Step 5: Generating Statistics Reports"
echo "=============================================="

for backend in vllm triton tgi; do
    echo "Generating stats for ${backend}..."

    # Kernel summary
    kubectl exec ${POD_NAME} -n ${NAMESPACE} -- bash -c "
        nsys stats \
            --report cuda_gpu_kern_sum \
            --format csv \
            --output /results/${backend}_kernel_stats \
            /results/${backend}_profile.nsys-rep
    " 2>/dev/null || echo "  Warning: Could not generate kernel stats"

    # Memory summary
    kubectl exec ${POD_NAME} -n ${NAMESPACE} -- bash -c "
        nsys stats \
            --report cuda_gpu_mem_time_sum \
            --format csv \
            --output /results/${backend}_memory_stats \
            /results/${backend}_profile.nsys-rep
    " 2>/dev/null || echo "  Warning: Could not generate memory stats"

    # CUDA API summary
    kubectl exec ${POD_NAME} -n ${NAMESPACE} -- bash -c "
        nsys stats \
            --report cuda_api_sum \
            --format csv \
            --output /results/${backend}_cuda_api_stats \
            /results/${backend}_profile.nsys-rep
    " 2>/dev/null || echo "  Warning: Could not generate CUDA API stats"
done

echo ""

# =============================================================================
# Step 6: List files in pod
# =============================================================================
echo "=============================================="
echo "  Step 6: Listing profile files in pod"
echo "=============================================="

kubectl exec ${POD_NAME} -n ${NAMESPACE} -- ls -lh /results/

echo ""

# =============================================================================
# Step 7: Copy results back to VM
# =============================================================================
echo "=============================================="
echo "  Step 7: Copying results to VM"
echo "=============================================="

echo "Copying from pod to: ${PROFILE_DIR}"

# Copy all profile files
for backend in vllm triton tgi; do
    echo "Copying ${backend} profiles..."

    # Copy .nsys-rep file
    kubectl cp ${NAMESPACE}/${POD_NAME}:/results/${backend}_profile.nsys-rep "${PROFILE_DIR}/${backend}_profile.nsys-rep" 2>/dev/null || \
        echo "  Warning: ${backend}_profile.nsys-rep not found"

    # Copy stats files
    for stat_type in kernel_stats memory_stats cuda_api_stats; do
        kubectl cp ${NAMESPACE}/${POD_NAME}:/results/${backend}_${stat_type}.csv "${PROFILE_DIR}/${backend}_${stat_type}.csv" 2>/dev/null || true
    done
done

echo ""

# =============================================================================
# Step 8: Generate summary report
# =============================================================================
echo "=============================================="
echo "  Step 8: Generating Summary Report"
echo "=============================================="

cat > "${PROFILE_DIR}/PROFILE_SUMMARY.md" << EOF
# Nsight Systems Profiling Results

## Generated: ${TIMESTAMP}

## Configuration
- **Model**: Mistral-7B-Instruct-v0.2
- **Backends**: vLLM, NVIDIA Triton (vLLM backend), HuggingFace TGI
- **Requests per profile**: ${PROFILE_REQUESTS}
- **Max tokens**: ${PROFILE_TOKENS}

## Profile Files

| Backend | Profile File | Size |
|---------|--------------|------|
EOF

for backend in vllm triton tgi; do
    if [ -f "${PROFILE_DIR}/${backend}_profile.nsys-rep" ]; then
        size=$(ls -lh "${PROFILE_DIR}/${backend}_profile.nsys-rep" | awk '{print $5}')
        echo "| ${backend^^} | ${backend}_profile.nsys-rep | ${size} |" >> "${PROFILE_DIR}/PROFILE_SUMMARY.md"
    fi
done

cat >> "${PROFILE_DIR}/PROFILE_SUMMARY.md" << EOF

## Statistics Files

| Backend | Kernel Stats | Memory Stats | CUDA API Stats |
|---------|--------------|--------------|----------------|
EOF

for backend in vllm triton tgi; do
    kernel="N/A"
    memory="N/A"
    cuda_api="N/A"
    [ -f "${PROFILE_DIR}/${backend}_kernel_stats.csv" ] && kernel="Yes"
    [ -f "${PROFILE_DIR}/${backend}_memory_stats.csv" ] && memory="Yes"
    [ -f "${PROFILE_DIR}/${backend}_cuda_api_stats.csv" ] && cuda_api="Yes"
    echo "| ${backend^^} | ${kernel} | ${memory} | ${cuda_api} |" >> "${PROFILE_DIR}/PROFILE_SUMMARY.md"
done

cat >> "${PROFILE_DIR}/PROFILE_SUMMARY.md" << EOF

## How to View Profiles

### Using Nsight Systems GUI (recommended)
\`\`\`bash
# On your local machine with nsys-ui installed
nsys-ui ${PROFILE_DIR}/vllm_profile.nsys-rep
nsys-ui ${PROFILE_DIR}/triton_profile.nsys-rep
nsys-ui ${PROFILE_DIR}/tgi_profile.nsys-rep
\`\`\`

### Using Command Line Stats
\`\`\`bash
# View kernel summary
nsys stats --report cuda_gpu_kern_sum ${PROFILE_DIR}/vllm_profile.nsys-rep

# View memory summary
nsys stats --report cuda_gpu_mem_time_sum ${PROFILE_DIR}/vllm_profile.nsys-rep

# Export to different formats
nsys stats --report cuda_gpu_kern_sum --format json --output ./stats ${PROFILE_DIR}/vllm_profile.nsys-rep
\`\`\`

## Key Metrics to Compare

1. **GPU Kernel Time**: Total time spent in GPU kernels
2. **Memory Transfer Time**: Time for CPU<->GPU data transfers
3. **CUDA API Overhead**: Time spent in CUDA API calls
4. **Kernel Launch Latency**: Time between kernel launches
5. **Memory Bandwidth Utilization**: How efficiently GPU memory is used

EOF

echo "Summary report created: ${PROFILE_DIR}/PROFILE_SUMMARY.md"
echo ""

# =============================================================================
# Step 9: Final Summary
# =============================================================================
echo "=============================================="
echo "  Profiling Complete!"
echo "=============================================="
echo ""
echo "Results saved to: ${PROFILE_DIR}"
echo ""
ls -la "${PROFILE_DIR}/"
echo ""
echo "Profile files (.nsys-rep) can be opened with Nsight Systems GUI"
echo ""
echo "To download to local machine:"
echo "  scp -r opc@<vm-ip>:${PROFILE_DIR} ./profiles/"
echo ""
echo "To view on Linux with nsys:"
echo "  nsys-ui ${PROFILE_DIR}/vllm_profile.nsys-rep"
echo ""

# Optional: Keep pod running or delete
read -p "Delete bench-client deployment? [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    kubectl delete deployment bench-client -n ${NAMESPACE}
    echo "bench-client deployment deleted"
else
    echo "bench-client deployment kept running"
    echo "To delete later: kubectl delete deployment bench-client -n ${NAMESPACE}"
fi
