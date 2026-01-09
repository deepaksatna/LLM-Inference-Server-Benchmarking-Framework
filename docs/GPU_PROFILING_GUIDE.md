# GPU Profiling Guide for LLM Inference Servers

## Complete Step-by-Step Instructions

This guide covers how to capture GPU profiling data for vLLM, NVIDIA Triton, and HuggingFace TGI inference servers running on Kubernetes.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Method 1: nvidia-smi GPU Monitoring (Recommended)](#method-1-nvidia-smi-gpu-monitoring)
3. [Method 2: Nsight Systems Client-Side Profiling](#method-2-nsight-systems-client-side-profiling)
4. [Method 3: Server-Side Nsys Profiling (Advanced)](#method-3-server-side-nsys-profiling)
5. [Analyzing Results](#analyzing-results)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### 1. Verify Kubernetes Access

```bash
# Check cluster connection
kubectl cluster-info

# Verify pods are running
kubectl get pods -n bench
```

Expected output:
```
NAME                             READY   STATUS    RESTARTS   AGE
bench-client-xxxxx               1/1     Running   0          1h
tgi-server-xxxxx                 1/1     Running   0          1h
triton-server-xxxxx              1/1     Running   0          1h
vllm-server-xxxxx                1/1     Running   0          1h
```

### 2. Verify GPU Access in Pods

```bash
# Check GPU in vLLM pod
VLLM_POD=$(kubectl get pod -l app=vllm-server -n bench -o jsonpath='{.items[0].metadata.name}')
kubectl exec ${VLLM_POD} -n bench -- nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv

# Check GPU in Triton pod
TRITON_POD=$(kubectl get pod -l app=triton-server -n bench -o jsonpath='{.items[0].metadata.name}')
kubectl exec ${TRITON_POD} -n bench -- nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv

# Check GPU in TGI pod
TGI_POD=$(kubectl get pod -l app=tgi-server -n bench -o jsonpath='{.items[0].metadata.name}')
kubectl exec ${TGI_POD} -n bench -- nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
```

### 3. Check nsys Availability

```bash
# Check if nsys is available in bench-client
BENCH_POD=$(kubectl get pod -l app=bench-client -n bench -o jsonpath='{.items[0].metadata.name}')
kubectl exec ${BENCH_POD} -n bench -- nsys --version

# Check if nsys is available in Triton (usually yes)
kubectl exec ${TRITON_POD} -n bench -- /usr/local/cuda/bin/nsys --version
```

---

## Method 1: nvidia-smi GPU Monitoring

**Best for**: Capturing real GPU metrics (SM utilization, memory, power, temperature) from inside the server pods during inference.

### Step 1: Set Up Pod Variables

```bash
# Get pod names
VLLM_POD=$(kubectl get pod -l app=vllm-server -n bench -o jsonpath='{.items[0].metadata.name}')
TRITON_POD=$(kubectl get pod -l app=triton-server -n bench -o jsonpath='{.items[0].metadata.name}')
TGI_POD=$(kubectl get pod -l app=tgi-server -n bench -o jsonpath='{.items[0].metadata.name}')

echo "vLLM Pod: ${VLLM_POD}"
echo "Triton Pod: ${TRITON_POD}"
echo "TGI Pod: ${TGI_POD}"
```

### Step 2: Profile vLLM Server

```bash
kubectl exec ${VLLM_POD} -n bench -- bash -c '
# Create results directory
mkdir -p /tmp/profiles

# Start GPU monitoring in background (1-second intervals)
nvidia-smi dmon -s pucvmet -d 1 > /tmp/profiles/vllm_gpu_dmon.log 2>&1 &
DMON_PID=$!
echo "Started nvidia-smi dmon (PID: $DMON_PID)"

# Wait for monitoring to start
sleep 2

# Generate inference load (5 requests, 100 tokens each)
echo "Making inference requests..."
for i in 1 2 3 4 5; do
    curl -s http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"mistralai/Mistral-7B-Instruct-v0.2\",\"prompt\":\"Explain how transformer attention mechanisms work in detail.\",\"max_tokens\":100}" > /dev/null
    echo "  Request $i complete"
done

# Wait for cool-down capture
sleep 2

# Stop monitoring
kill $DMON_PID 2>/dev/null
echo "GPU monitoring stopped"

# Display results
echo ""
echo "=== GPU Metrics During Inference ==="
cat /tmp/profiles/vllm_gpu_dmon.log
'
```

### Step 3: Profile Triton Server

```bash
kubectl exec ${TRITON_POD} -n bench -- bash -c '
mkdir -p /tmp/profiles

nvidia-smi dmon -s pucvmet -d 1 > /tmp/profiles/triton_gpu_dmon.log 2>&1 &
DMON_PID=$!
echo "Started nvidia-smi dmon (PID: $DMON_PID)"

sleep 2

echo "Making inference requests..."
for i in 1 2 3 4 5; do
    curl -s http://localhost:8000/v2/models/mistral/generate \
        -H "Content-Type: application/json" \
        -d "{\"text_input\":\"Explain how transformer attention mechanisms work in detail.\",\"parameters\":{\"max_tokens\":100}}" > /dev/null
    echo "  Request $i complete"
done

sleep 2
kill $DMON_PID 2>/dev/null

echo ""
echo "=== GPU Metrics During Inference ==="
cat /tmp/profiles/triton_gpu_dmon.log
'
```

### Step 4: Profile TGI Server

```bash
kubectl exec ${TGI_POD} -n bench -- bash -c '
mkdir -p /tmp/profiles

nvidia-smi dmon -s pucvmet -d 1 > /tmp/profiles/tgi_gpu_dmon.log 2>&1 &
DMON_PID=$!
echo "Started nvidia-smi dmon (PID: $DMON_PID)"

sleep 2

echo "Making inference requests..."
for i in 1 2 3 4 5; do
    curl -s http://localhost:8000/generate \
        -H "Content-Type: application/json" \
        -d "{\"inputs\":\"Explain how transformer attention mechanisms work in detail.\",\"parameters\":{\"max_new_tokens\":100}}" > /dev/null
    echo "  Request $i complete"
done

sleep 2
kill $DMON_PID 2>/dev/null

echo ""
echo "=== GPU Metrics During Inference ==="
cat /tmp/profiles/tgi_gpu_dmon.log
'
```

### Step 5: Copy Results Locally

```bash
# Create local directory
LOCAL_DIR="./gpu_profiles_$(date +%Y%m%d)"
mkdir -p "$LOCAL_DIR"

# Copy from each pod
kubectl cp bench/${VLLM_POD}:/tmp/profiles/vllm_gpu_dmon.log "$LOCAL_DIR/vllm_gpu_dmon.log"
kubectl cp bench/${TRITON_POD}:/tmp/profiles/triton_gpu_dmon.log "$LOCAL_DIR/triton_gpu_dmon.log"
kubectl cp bench/${TGI_POD}:/tmp/profiles/tgi_gpu_dmon.log "$LOCAL_DIR/tgi_gpu_dmon.log"

echo "Files saved to: $LOCAL_DIR"
ls -la "$LOCAL_DIR"
```

### Understanding nvidia-smi dmon Output

```
# Column descriptions:
# gpu  - GPU index
# pwr  - Power draw (Watts)
# gtemp - GPU temperature (Celsius)
# mtemp - Memory temperature (Celsius)
# sm   - Streaming Multiprocessor utilization (%)
# mem  - Memory bandwidth utilization (%)
# enc  - Encoder utilization (%)
# dec  - Decoder utilization (%)
# mclk - Memory clock (MHz)
# pclk - GPU clock (MHz)
```

---

## Method 2: Nsight Systems Client-Side Profiling

**Best for**: Capturing request timing, OS runtime analysis, and thread activity from the client making inference requests.

### Step 1: Set Up Bench-Client Pod

```bash
BENCH_POD=$(kubectl get pod -l app=bench-client -n bench -o jsonpath='{.items[0].metadata.name}')
echo "Bench-client Pod: ${BENCH_POD}"

# Verify nsys is available
kubectl exec ${BENCH_POD} -n bench -- nsys --version
```

### Step 2: Create Profiling Script in Pod

```bash
kubectl exec ${BENCH_POD} -n bench -- bash -c '
cat > /tmp/profile_backend.py << "EOFPY"
import requests
import time
import sys
import concurrent.futures

def make_requests(backend, url, num_requests, max_tokens):
    prompt = "Explain the complete architecture of transformer neural networks including multi-head attention, feed-forward layers, layer normalization, and positional encoding in detail."

    print(f"[{backend}] Starting {num_requests} requests...")

    def single_request(i):
        start = time.time()
        try:
            if backend == "vllm":
                resp = requests.post(f"{url}/v1/completions", json={
                    "model": "mistralai/Mistral-7B-Instruct-v0.2",
                    "prompt": prompt,
                    "max_tokens": max_tokens
                }, timeout=180)
            elif backend == "triton":
                resp = requests.post(f"{url}/v2/models/mistral/generate", json={
                    "text_input": prompt,
                    "parameters": {"max_tokens": max_tokens}
                }, timeout=180)
            elif backend == "tgi":
                resp = requests.post(f"{url}/generate", json={
                    "inputs": prompt,
                    "parameters": {"max_new_tokens": max_tokens}
                }, timeout=180)

            latency = time.time() - start
            return {"success": resp.status_code == 200, "latency": latency}
        except Exception as e:
            return {"success": False, "latency": 0, "error": str(e)}

    # Run concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(single_request, range(num_requests)))

    successful = sum(1 for r in results if r["success"])
    avg_latency = sum(r["latency"] for r in results if r["success"]) / max(successful, 1)
    print(f"[{backend}] Completed: {successful}/{num_requests}, avg latency: {avg_latency:.2f}s")
    return results

if __name__ == "__main__":
    backend = sys.argv[1]
    urls = {
        "vllm": "http://vllm-service.bench.svc.cluster.local:8000",
        "triton": "http://triton-service.bench.svc.cluster.local:8000",
        "tgi": "http://tgi-service.bench.svc.cluster.local:8000"
    }
    make_requests(backend, urls[backend], num_requests=15, max_tokens=100)
EOFPY

echo "Profiling script created"
'
```

### Step 3: Run nsys Profile for Each Backend

```bash
# Create results directory
kubectl exec ${BENCH_POD} -n bench -- mkdir -p /results/nsys_profiles

# Profile vLLM
echo "Profiling vLLM..."
kubectl exec ${BENCH_POD} -n bench -- bash -c '
nsys profile \
    --output=/results/nsys_profiles/vllm_inference_profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    --stats=true \
    python3 /tmp/profile_backend.py vllm
'

# Profile Triton
echo "Profiling Triton..."
kubectl exec ${BENCH_POD} -n bench -- bash -c '
nsys profile \
    --output=/results/nsys_profiles/triton_inference_profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    --stats=true \
    python3 /tmp/profile_backend.py triton
'

# Profile TGI
echo "Profiling TGI..."
kubectl exec ${BENCH_POD} -n bench -- bash -c '
nsys profile \
    --output=/results/nsys_profiles/tgi_inference_profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    --stats=true \
    python3 /tmp/profile_backend.py tgi
'
```

### Step 4: Copy nsys Profiles Locally

```bash
LOCAL_DIR="./nsys_profiles_$(date +%Y%m%d)"
mkdir -p "$LOCAL_DIR"

kubectl cp bench/${BENCH_POD}:/results/nsys_profiles "$LOCAL_DIR"

echo "Profiles saved to: $LOCAL_DIR"
ls -la "$LOCAL_DIR"
```

---

## Method 3: Server-Side Nsys Profiling (Advanced)

**Best for**: Capturing actual GPU kernel data from the inference server process.

**Note**: This requires restarting the server with nsys wrapping the process.

### Option A: Use Triton (Has nsys Pre-installed)

```bash
# Step 1: Scale down existing Triton deployment
kubectl scale deployment triton-server -n bench --replicas=0
sleep 10

# Step 2: Create profiling pod
cat << 'EOF' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: triton-gpu-profiler
  namespace: bench
spec:
  restartPolicy: Never
  serviceAccountName: bench-sa
  imagePullSecrets:
    - name: ocirsecret

  containers:
  - name: triton-profiler
    image: fra.ocir.io/frntrd2vyxvi/models:llm-triton-offline-v1  # Use your Triton image
    command: ["/bin/bash", "-c"]
    args:
      - |
        echo "Starting Triton with nsys GPU profiling..."
        mkdir -p /results

        /usr/local/cuda/bin/nsys profile \
            --output=/results/triton_server_gpu_kernels \
            --trace=cuda,nvtx,cudnn,cublas \
            --cuda-memory-usage=true \
            --force-overwrite=true \
            --duration=150 \
            tritonserver \
                --model-repository=/model-repository \
                --http-port=8000 \
                --grpc-port=8001 \
                --metrics-port=8002 \
                --strict-model-config=false &

        TRITON_PID=$!

        # Wait for server ready
        for i in {1..180}; do
            if curl -s http://localhost:8000/v2/health/ready > /dev/null 2>&1; then
                echo "Triton ready!"
                break
            fi
            sleep 1
        done

        # Generate load
        for i in {1..10}; do
            curl -s http://localhost:8000/v2/models/mistral/generate \
                -H "Content-Type: application/json" \
                -d '{"text_input":"Explain transformers.","parameters":{"max_tokens":100}}' > /dev/null
            echo "Request $i"
        done

        wait $TRITON_PID || true

        echo "Profile complete!"
        ls -lh /results/
        sleep 7200

    resources:
      requests:
        nvidia.com/gpu: "1"
      limits:
        nvidia.com/gpu: "1"

    volumeMounts:
    - name: results
      mountPath: /results
    - name: dshm
      mountPath: /dev/shm

  volumes:
  - name: results
    emptyDir: {}
  - name: dshm
    emptyDir:
      medium: Memory
      sizeLimit: "16Gi"

  nodeSelector:
    nvidia.com/gpu: "true"

  tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
EOF

# Step 3: Wait for profiling to complete
kubectl logs -f triton-gpu-profiler -n bench

# Step 4: Copy results
kubectl cp bench/triton-gpu-profiler:/results/triton_server_gpu_kernels.nsys-rep ./triton_server_gpu_kernels.nsys-rep

# Step 5: Cleanup and restore
kubectl delete pod triton-gpu-profiler -n bench
kubectl scale deployment triton-server -n bench --replicas=1
```

### Option B: For vLLM/TGI (Requires nsys Installation)

For vLLM and TGI, you need to either:
1. Build custom images with nsys pre-installed
2. Install nsys at runtime (slower startup)

Example Dockerfile addition:
```dockerfile
# Add to your vLLM/TGI Dockerfile
RUN apt-get update && apt-get install -y wget && \
    wget https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/nsight-systems-cli-2024.4.2_2024.4.2.133-1_amd64.deb && \
    dpkg -i nsight-systems-cli-*.deb || apt-get install -f -y && \
    rm nsight-systems-cli-*.deb
```

---

## Analyzing Results

### View nvidia-smi dmon Logs

```bash
# View GPU metrics
cat vllm_gpu_dmon.log

# Key metrics to look for:
# - sm: Should be 90-99% during inference (GPU compute usage)
# - mem: Should be ~100% during inference (memory bandwidth)
# - pwr: Active power draw (e.g., 150W)
# - gtemp: Temperature during inference
```

### View nsys Profiles in GUI

```bash
# Open Nsight Systems GUI
nsys-ui vllm_inference_profile.nsys-rep
```

### Generate nsys Stats via CLI

```bash
# OS Runtime Summary
nsys stats --report osrt_sum vllm_inference_profile.nsys-rep

# CUDA Kernel Summary (if available)
nsys stats --report cuda_gpu_kern_sum vllm_inference_profile.nsys-rep

# Export to CSV
nsys stats --report osrt_sum --format csv --output vllm_stats vllm_inference_profile.nsys-rep
```

---

## Troubleshooting

### Issue: "nsys not found"

**Solution**: Only Triton image has nsys pre-installed. For vLLM/TGI, use nvidia-smi dmon method or build custom image with nsys.

### Issue: "does not contain CUDA kernel data"

**Cause**: nsys profiled the client process (making HTTP requests), not the server process (running inference).

**Solution**: Use nvidia-smi dmon for GPU metrics, or restart server with nsys wrapping (Method 3).

### Issue: Pod fails to start

**Check**: Image name, GPU resources, node selector
```bash
kubectl describe pod <pod-name> -n bench | tail -30
```

### Issue: Server not responding

**Check**: Wait longer for model loading (2-3 minutes for 7B model)
```bash
kubectl logs <pod-name> -n bench --tail=50
```

---

## Quick Reference Commands

```bash
# Get pod names
VLLM_POD=$(kubectl get pod -l app=vllm-server -n bench -o jsonpath='{.items[0].metadata.name}')
TRITON_POD=$(kubectl get pod -l app=triton-server -n bench -o jsonpath='{.items[0].metadata.name}')
TGI_POD=$(kubectl get pod -l app=tgi-server -n bench -o jsonpath='{.items[0].metadata.name}')
BENCH_POD=$(kubectl get pod -l app=bench-client -n bench -o jsonpath='{.items[0].metadata.name}')

# Check GPU status
kubectl exec ${VLLM_POD} -n bench -- nvidia-smi

# Copy file from pod
kubectl cp bench/${POD_NAME}:/path/in/pod ./local/path

# Stream logs
kubectl logs -f ${POD_NAME} -n bench

# Execute command in pod
kubectl exec ${POD_NAME} -n bench -- <command>
```

---

## Summary

| Method | What it Captures | Difficulty | Best For |
|--------|------------------|------------|----------|
| nvidia-smi dmon | Real GPU metrics (SM%, power, temp) | Easy | Server-side GPU utilization |
| nsys Client-side | Request timing, OS runtime | Easy | End-to-end latency analysis |
| nsys Server-side | GPU kernels, CUDA API | Hard | Deep GPU kernel analysis |

**Recommended Approach**: Use **Method 1 (nvidia-smi dmon)** for GPU metrics + **Method 2 (nsys client-side)** for timing profiles.

---

*Guide created: January 8, 2026*
