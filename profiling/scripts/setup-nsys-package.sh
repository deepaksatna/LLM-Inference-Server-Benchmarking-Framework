#!/bin/bash
#
# Setup Nsight Systems Package on Kubernetes GPU Nodes
#
# This script copies the nsys .deb package to GPU nodes so that profiling
# jobs can mount it as a hostPath volume.
#
# Prerequisites:
#   - kubectl configured with cluster access
#   - nsys .deb package available locally
#
# Usage:
#   ./setup-nsys-package.sh <path-to-nsys-deb> <namespace>
#
# Example:
#   ./setup-nsys-package.sh ./nsight-systems-cli-2025.6.1.deb bench

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
NSYS_PACKAGE="${1:-./offline-packages/nsight-systems-cli-2025.6.1.deb}"
NAMESPACE="${2:-bench}"
TARGET_PATH="/tmp/nsight-systems-cli-2025.6.1.deb"

echo "=============================================="
echo "  Nsight Systems Package Setup for Kubernetes"
echo "=============================================="
echo ""

# Check if nsys package exists
if [ ! -f "$NSYS_PACKAGE" ]; then
    echo -e "${RED}ERROR: nsys package not found at $NSYS_PACKAGE${NC}"
    echo ""
    echo "Please provide the path to the nsight-systems-cli .deb package"
    echo "Download from: https://developer.nvidia.com/nsight-systems"
    exit 1
fi

PACKAGE_SIZE=$(ls -lh "$NSYS_PACKAGE" | awk '{print $5}')
echo "Package: $NSYS_PACKAGE ($PACKAGE_SIZE)"
echo "Target: $TARGET_PATH on GPU nodes"
echo "Namespace: $NAMESPACE"
echo ""

# Get list of GPU nodes
echo "Finding GPU nodes..."
GPU_NODES=$(kubectl get nodes -l nvidia.com/gpu=true -o jsonpath='{.items[*].metadata.name}')

if [ -z "$GPU_NODES" ]; then
    echo -e "${YELLOW}WARNING: No nodes with label nvidia.com/gpu=true found${NC}"
    echo "Trying to find nodes with GPU resources..."
    GPU_NODES=$(kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}{" "}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}' | grep -v "^[^ ]* $" | awk '{print $1}')
fi

if [ -z "$GPU_NODES" ]; then
    echo -e "${RED}ERROR: No GPU nodes found in cluster${NC}"
    exit 1
fi

echo "Found GPU nodes: $GPU_NODES"
echo ""

# Create a temporary pod to copy the file to each node
copy_to_node() {
    local NODE=$1
    echo "Copying to node: $NODE"

    # Create a temporary pod on the specific node
    POD_NAME="nsys-copy-$(echo $NODE | tr '.' '-' | cut -c1-20)-$$"

    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: $POD_NAME
  namespace: $NAMESPACE
spec:
  nodeName: $NODE
  restartPolicy: Never
  containers:
  - name: copy
    image: busybox
    command: ["sleep", "300"]
    volumeMounts:
    - name: host-tmp
      mountPath: /host-tmp
  volumes:
  - name: host-tmp
    hostPath:
      path: /tmp
      type: Directory
  tolerations:
  - operator: Exists
EOF

    # Wait for pod to be ready
    echo "  Waiting for copy pod to be ready..."
    kubectl wait --for=condition=Ready pod/$POD_NAME -n $NAMESPACE --timeout=60s 2>/dev/null || true

    # Copy the package
    echo "  Copying nsys package..."
    kubectl cp "$NSYS_PACKAGE" "$NAMESPACE/$POD_NAME:/host-tmp/nsight-systems-cli-2025.6.1.deb"

    # Verify copy
    COPIED_SIZE=$(kubectl exec -n $NAMESPACE $POD_NAME -- ls -l /host-tmp/nsight-systems-cli-2025.6.1.deb 2>/dev/null | awk '{print $5}')

    if [ -n "$COPIED_SIZE" ]; then
        echo -e "  ${GREEN}SUCCESS: Package copied ($COPIED_SIZE bytes)${NC}"
    else
        echo -e "  ${RED}WARNING: Copy may have failed${NC}"
    fi

    # Cleanup
    kubectl delete pod $POD_NAME -n $NAMESPACE --force --grace-period=0 2>/dev/null || true
    echo ""
}

# Copy to each GPU node
for NODE in $GPU_NODES; do
    copy_to_node $NODE
done

echo "=============================================="
echo -e "${GREEN}  Setup Complete${NC}"
echo "=============================================="
echo ""
echo "The nsys package is now available at $TARGET_PATH on all GPU nodes."
echo ""
echo "You can now deploy profiling jobs:"
echo "  kubectl apply -f jobs/vllm-nsys-profiler.yaml"
echo "  kubectl apply -f jobs/triton-nsys-profiler.yaml"
echo "  kubectl apply -f jobs/tgi-nsys-profiler.yaml"
