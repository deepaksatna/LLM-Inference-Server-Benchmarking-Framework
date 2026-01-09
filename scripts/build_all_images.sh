#!/usr/bin/env bash
# Build all LLM inference images with pre-downloaded models
# Usage: ./build_all_images.sh [--skip-inference] [--only-client]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

REGISTRY="fra.ocir.io/frntrd2vyxvi/models"
VERSION="v1"

SKIP_INFERENCE=false
ONLY_CLIENT=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --skip-inference) SKIP_INFERENCE=true ;;
        --only-client) ONLY_CLIENT=true ;;
    esac
done

echo "======================================================="
echo "  Building LLM Inference Images (Offline)"
echo "======================================================="
echo ""
echo "Registry: ${REGISTRY}"
echo "Version:  ${VERSION}"
echo "Project:  ${PROJECT_ROOT}"
echo ""

# Check HF_TOKEN for inference images
if [ "$ONLY_CLIENT" = false ] && [ "$SKIP_INFERENCE" = false ]; then
    if [ -z "${HF_TOKEN:-}" ]; then
        echo "ERROR: HF_TOKEN environment variable not set"
        echo ""
        echo "Get your token from: https://huggingface.co/settings/tokens"
        echo "Then run: export HF_TOKEN='hf_xxxxxxxxxx'"
        echo ""
        echo "Or build only the benchmark client:"
        echo "  ./build_all_images.sh --only-client"
        exit 1
    fi
    echo "HF_TOKEN: Set (${#HF_TOKEN} characters)"
fi

echo ""

build_image() {
    local name=$1
    local dockerfile=$2
    local tag="${REGISTRY}:${name}-${VERSION}"
    local extra_args="${3:-}"

    echo "======================================================="
    echo "  Building: ${name}"
    echo "  Tag: ${tag}"
    echo "======================================================="
    echo ""

    if [ -n "$extra_args" ]; then
        docker build $extra_args -t "$tag" -f "$dockerfile" .
    else
        docker build -t "$tag" -f "$dockerfile" .
    fi

    echo ""
    echo "Built: ${tag}"
    echo ""
}

# Build benchmark client (always)
echo "[1/?] Building Benchmark Client..."
build_image "llm-bench-client" "docker/bench-client/Dockerfile"

if [ "$ONLY_CLIENT" = true ]; then
    echo "======================================================="
    echo "  Build Complete (Client Only)"
    echo "======================================================="
    docker images | grep "llm-bench-client"
    exit 0
fi

if [ "$SKIP_INFERENCE" = false ]; then
    # Build vLLM
    echo "[2/4] Building vLLM (with Mistral-7B model)..."
    echo "This will take 10-20 minutes to download the model..."
    build_image "llm-vllm-offline" "docker/vllm/Dockerfile" "--build-arg HF_TOKEN=${HF_TOKEN}"

    # Build Triton
    echo "[3/4] Building Triton (with Mistral-7B model)..."
    echo "This will take 10-20 minutes to download the model..."
    build_image "llm-triton-offline" "docker/triton/Dockerfile" "--build-arg HF_TOKEN=${HF_TOKEN}"

    # Build TGI
    echo "[4/4] Building TGI (with Mistral-7B model)..."
    echo "This will take 10-20 minutes to download the model..."
    build_image "llm-tgi-offline" "docker/tgi/Dockerfile" "--build-arg HF_TOKEN=${HF_TOKEN}"
fi

echo ""
echo "======================================================="
echo "  Build Complete!"
echo "======================================================="
echo ""
echo "Images created:"
docker images | grep -E "(llm-vllm|llm-triton|llm-tgi|llm-bench)"
echo ""
echo "Total size:"
docker images | grep -E "(llm-vllm|llm-triton|llm-tgi|llm-bench)" | awk '{sum+=$7} END {print sum " GB (approx)"}'
echo ""
echo "Next steps:"
echo "  1. Push to OCIR: ./scripts/push_all_images.sh"
echo "  2. Deploy to OKE: ./scripts/deploy_backend.sh vllm --wait"
echo ""
