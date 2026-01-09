# Quick Start Guide

Get up and running with LLM inference benchmarking in 15 minutes.

---

## Prerequisites

- Kubernetes cluster with GPU nodes
- kubectl configured
- Docker installed
- 24GB+ GPU VRAM (A10, A100, H100, etc.)

---

## Step 1: Clone and Configure (2 min)

```bash
# Clone the repository
git clone https://github.com/YOUR_ORG/llm-inference-benchmark.git
cd llm-inference-benchmark

# Set your container registry
export REGISTRY="your-registry.com/namespace"

# Example for different registries:
# OCI:   fra.ocir.io/namespace/models
# AWS:   123456789.dkr.ecr.region.amazonaws.com
# GCP:   gcr.io/project-id
# Azure: myregistry.azurecr.io
```

---

## Step 2: Build Docker Images (5 min)

```bash
# Build all three inference server images
./scripts/build_all_images.sh

# This builds:
# - vLLM server
# - NVIDIA Triton server
# - HuggingFace TGI server
# - Benchmark client
```

---

## Step 3: Push Images (2 min)

```bash
# Push to your registry
./scripts/push_all_images.sh

# Verify images are pushed
docker images | grep llm
```

---

## Step 4: Deploy to Kubernetes (3 min)

```bash
# Create namespace and common resources
kubectl apply -f k8s/common/namespace.yaml
kubectl apply -f k8s/common/model-pvc.yaml

# Create image pull secret (if using private registry)
kubectl create secret docker-registry ocirsecret \
    --docker-server=$REGISTRY \
    --docker-username=YOUR_USER \
    --docker-password=YOUR_TOKEN \
    -n bench

# Deploy inference servers
kubectl apply -f k8s/vllm/deployment.yaml
kubectl apply -f k8s/triton/deployment.yaml
kubectl apply -f k8s/tgi/deployment.yaml

# Wait for pods to be ready (model loading takes 2-5 min)
kubectl get pods -n bench -w
```

---

## Step 5: Run Benchmark (3 min)

```bash
# Deploy benchmark client
kubectl apply -f k8s/bench-client/deployment.yaml

# Run benchmark
./scripts/run_full_benchmark.sh

# Or run individual server benchmark
./scripts/run_benchmark.sh vllm
```

---

## Step 6: View Results

```bash
# Generate visualizations
python scripts/generate_visualizations.py

# Results are in:
# - results/benchmarks/       # JSON data
# - results/plots/            # PNG visualizations
```

---

## Quick Verification

```bash
# Check all pods are running
kubectl get pods -n bench

# Expected output:
# NAME                           READY   STATUS
# vllm-server-xxx                1/1     Running
# triton-server-xxx              1/1     Running
# tgi-server-xxx                 1/1     Running
# bench-client-xxx               1/1     Running

# Test vLLM endpoint
kubectl exec -it deploy/bench-client -n bench -- \
    curl -s http://vllm-server:8000/v1/models

# Test Triton endpoint
kubectl exec -it deploy/bench-client -n bench -- \
    curl -s http://triton-server:8000/v2/health/ready

# Test TGI endpoint
kubectl exec -it deploy/bench-client -n bench -- \
    curl -s http://tgi-server:8000/health
```

---

## Common Issues

### Pods stuck in Pending
```bash
# Check events
kubectl describe pod <pod-name> -n bench

# Common causes:
# - No GPU nodes available
# - Insufficient memory
# - Image pull errors
```

### Model loading timeout
```bash
# Increase timeout in deployment
# Or use pre-cached model images
```

### Out of memory
```bash
# Reduce max-model-len in deployment
# Or use smaller model
```

---

## Next Steps

1. **Customize benchmarks**: Edit `benchmarks/configs/benchmark_matrix.yaml`
2. **Add GPU profiling**: See `profiling/README.md`
3. **Try different models**: Update model name in deployments
4. **Scale testing**: Increase concurrency levels

---

## One-Liner Deploy (for the impatient)

```bash
export REGISTRY="your-registry.com/namespace" && \
./scripts/build_all_images.sh && \
./scripts/push_all_images.sh && \
kubectl apply -f k8s/common/ && \
kubectl apply -f k8s/vllm/ && \
kubectl apply -f k8s/triton/ && \
kubectl apply -f k8s/tgi/ && \
kubectl apply -f k8s/bench-client/ && \
echo "Deployed! Run: kubectl get pods -n bench -w"
```
