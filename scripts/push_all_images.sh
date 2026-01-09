#!/usr/bin/env bash
# Push all LLM inference images to OCIR
# Usage: ./push_all_images.sh [--only-client]
set -euo pipefail

REGISTRY="fra.ocir.io/frntrd2vyxvi/models"
VERSION="v1"

ONLY_CLIENT=false
[ "${1:-}" = "--only-client" ] && ONLY_CLIENT=true

echo "======================================================="
echo "  Pushing Images to OCIR"
echo "======================================================="
echo ""
echo "Registry: ${REGISTRY}"
echo ""

# Check OCIR login
echo "Checking OCIR authentication..."
if ! docker pull ${REGISTRY}:test 2>&1 | grep -q "manifest unknown\|not found"; then
    # If we don't get "not found", we might not be logged in
    echo "Verifying OCIR login..."
fi

push_image() {
    local name=$1
    local tag="${REGISTRY}:${name}-${VERSION}"

    echo ""
    echo "Pushing: ${tag}"
    echo "Size: $(docker images ${tag} --format '{{.Size}}')"
    echo ""

    docker push "$tag"

    echo "Pushed: ${tag}"
}

# Push benchmark client
echo "[1/?] Pushing Benchmark Client..."
push_image "llm-bench-client"

if [ "$ONLY_CLIENT" = false ]; then
    # Push vLLM
    echo "[2/4] Pushing vLLM..."
    push_image "llm-vllm-offline"

    # Push Triton
    echo "[3/4] Pushing Triton..."
    push_image "llm-triton-offline"

    # Push TGI
    echo "[4/4] Pushing TGI..."
    push_image "llm-tgi-offline"
fi

echo ""
echo "======================================================="
echo "  Push Complete!"
echo "======================================================="
echo ""
echo "Images in OCIR:"
if [ "$ONLY_CLIENT" = true ]; then
    echo "  ${REGISTRY}:llm-bench-client-${VERSION}"
else
    echo "  ${REGISTRY}:llm-vllm-offline-${VERSION}"
    echo "  ${REGISTRY}:llm-triton-offline-${VERSION}"
    echo "  ${REGISTRY}:llm-tgi-offline-${VERSION}"
    echo "  ${REGISTRY}:llm-bench-client-${VERSION}"
fi
echo ""
echo "Next steps:"
echo "  1. Create namespace: kubectl apply -f k8s/common/namespace.yaml"
echo "  2. Create OCIR secret: kubectl create secret docker-registry ocirsecret ..."
echo "  3. Deploy backend: ./scripts/deploy_backend.sh vllm --wait"
echo ""
