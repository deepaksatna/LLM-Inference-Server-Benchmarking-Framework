#!/usr/bin/env bash
# Build benchmark client Docker image
# Usage: ./build.sh [image-tag]

set -euo pipefail

IMAGE_TAG="${1:-fra.ocir.io/frntrd2vyxvi/models:llm-bench-client-v1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "Building benchmark client image: ${IMAGE_TAG}"
echo "Context: ${PROJECT_ROOT}"

docker build \
    -t "${IMAGE_TAG}" \
    -f "${SCRIPT_DIR}/Dockerfile" \
    "${PROJECT_ROOT}"

echo ""
echo "Build complete: ${IMAGE_TAG}"
echo ""
echo "Push to OCIR:"
echo "  docker push ${IMAGE_TAG}"
