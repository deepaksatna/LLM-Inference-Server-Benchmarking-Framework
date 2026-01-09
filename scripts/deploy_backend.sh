#!/usr/bin/env bash
# Deploy inference backend to OKE
# Usage: ./deploy_backend.sh <backend> [--wait]
#   backend: vllm, triton, tgi, all
#   --wait: Wait for deployment to be ready

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
K8S_DIR="${PROJECT_ROOT}/k8s"

BACKEND="${1:-}"
WAIT_FLAG="${2:-}"

if [ -z "$BACKEND" ]; then
    echo "Usage: ./deploy_backend.sh <backend> [--wait]"
    echo ""
    echo "Backends:"
    echo "  vllm    - vLLM inference server"
    echo "  triton  - NVIDIA Triton inference server"
    echo "  tgi     - HuggingFace Text Generation Inference"
    echo "  all     - Deploy all backends"
    echo ""
    echo "Options:"
    echo "  --wait  - Wait for deployment to be ready"
    exit 1
fi

deploy_backend() {
    local backend=$1

    echo "======================================================="
    echo "  Deploying: ${backend}"
    echo "======================================================="

    # Apply common resources first
    kubectl apply -f "${K8S_DIR}/common/namespace.yaml"

    # Check for OCIR secret
    if ! kubectl get secret ocirsecret -n bench &> /dev/null; then
        echo "WARNING: OCIR secret 'ocirsecret' not found in bench namespace"
        echo "Create it with:"
        echo "  kubectl create secret docker-registry ocirsecret \\"
        echo "    --docker-server=fra.ocir.io \\"
        echo "    --docker-username='frntrd2vyxvi/oracleidentitycloudservice/<email>' \\"
        echo "    --docker-password='<auth-token>' \\"
        echo "    -n bench"
    fi

    # Deploy backend
    kubectl apply -f "${K8S_DIR}/${backend}/"

    echo ""
    echo "Deployment applied successfully."
}

wait_for_ready() {
    local backend=$1
    local deployment_name

    case $backend in
        vllm)   deployment_name="vllm-server" ;;
        triton) deployment_name="triton-server" ;;
        tgi)    deployment_name="tgi-server" ;;
        *)      return ;;
    esac

    echo ""
    echo "Waiting for ${deployment_name} to be ready..."
    kubectl rollout status deployment/${deployment_name} -n bench --timeout=600s

    echo ""
    echo "Checking pods:"
    kubectl get pods -n bench -l backend=${backend}

    echo ""
    echo "Checking services:"
    kubectl get svc -n bench | grep -E "(NAME|${backend})"
}

# Deploy based on selection
case $BACKEND in
    vllm|triton|tgi)
        deploy_backend "$BACKEND"
        if [ "$WAIT_FLAG" == "--wait" ]; then
            wait_for_ready "$BACKEND"
        fi
        ;;
    all)
        for b in vllm triton tgi; do
            deploy_backend "$b"
        done
        if [ "$WAIT_FLAG" == "--wait" ]; then
            for b in vllm triton tgi; do
                wait_for_ready "$b"
            done
        fi
        ;;
    *)
        echo "ERROR: Unknown backend: $BACKEND"
        echo "Supported: vllm, triton, tgi, all"
        exit 1
        ;;
esac

echo ""
echo "======================================================="
echo "  Deployment Complete"
echo "======================================================="
echo ""
echo "To check status:"
echo "  kubectl get pods -n bench"
echo "  kubectl get svc -n bench"
echo ""
echo "To view logs:"
echo "  kubectl logs -f deployment/${BACKEND}-server -n bench"
echo ""
