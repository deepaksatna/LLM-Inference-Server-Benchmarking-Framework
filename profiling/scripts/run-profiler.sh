#!/bin/bash
#
# Run Nsight Systems Profiler for LLM Inference Servers
#
# This script handles the complete profiling workflow:
#   1. Scale down existing deployment
#   2. Deploy profiling job
#   3. Wait for profiling to complete
#   4. Extract profile to local machine
#   5. Restore original deployment
#
# Usage:
#   ./run-profiler.sh <server> [output-dir]
#
# Examples:
#   ./run-profiler.sh vllm ./profiles
#   ./run-profiler.sh triton ./profiles
#   ./run-profiler.sh tgi ./profiles
#   ./run-profiler.sh all ./profiles

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
SERVER="${1:-vllm}"
OUTPUT_DIR="${2:-./profiles}"
NAMESPACE="bench"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOBS_DIR="$SCRIPT_DIR/../jobs"

# Server configurations
declare -A DEPLOYMENT_NAMES=(
    ["vllm"]="vllm-server"
    ["triton"]="triton-server"
    ["tgi"]="tgi-server"
)

declare -A JOB_NAMES=(
    ["vllm"]="vllm-nsys-profiler"
    ["triton"]="triton-nsys-profiler"
    ["tgi"]="tgi-nsys-profiler"
)

declare -A JOB_FILES=(
    ["vllm"]="vllm-nsys-profiler.yaml"
    ["triton"]="triton-nsys-profiler.yaml"
    ["tgi"]="tgi-nsys-profiler.yaml"
)

declare -A PROFILE_PATHS=(
    ["vllm"]="/results/vllm_gpu_timeline.nsys-rep"
    ["triton"]="/results/triton_gpu_timeline.nsys-rep"
    ["tgi"]="/results/tgi_gpu_timeline.nsys-rep"
)

# Functions
print_header() {
    echo ""
    echo -e "${BLUE}=============================================="
    echo "  $1"
    echo -e "==============================================${NC}"
    echo ""
}

print_step() {
    echo -e "${YELLOW}>>> $1${NC}"
}

print_success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

print_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

wait_for_pod_ready() {
    local JOB_NAME=$1
    local TIMEOUT=${2:-600}

    print_step "Waiting for profiler pod to be ready (timeout: ${TIMEOUT}s)..."

    for i in $(seq 1 $((TIMEOUT / 5))); do
        POD_NAME=$(kubectl get pods -n $NAMESPACE -l job-name=$JOB_NAME -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
        if [ -n "$POD_NAME" ]; then
            POD_STATUS=$(kubectl get pod $POD_NAME -n $NAMESPACE -o jsonpath='{.status.phase}' 2>/dev/null)
            if [ "$POD_STATUS" = "Running" ]; then
                echo "Pod $POD_NAME is running"
                return 0
            elif [ "$POD_STATUS" = "Succeeded" ] || [ "$POD_STATUS" = "Failed" ]; then
                echo "Pod $POD_NAME completed with status: $POD_STATUS"
                return 0
            fi
        fi
        echo "  Waiting... ($i)"
        sleep 5
    done

    print_error "Timeout waiting for pod"
    return 1
}

wait_for_profile() {
    local POD_NAME=$1
    local PROFILE_PATH=$2
    local TIMEOUT=${3:-900}

    print_step "Waiting for profile generation (timeout: ${TIMEOUT}s)..."

    for i in $(seq 1 $((TIMEOUT / 10))); do
        # Check if profile file exists
        if kubectl exec -n $NAMESPACE $POD_NAME -- ls -la $PROFILE_PATH 2>/dev/null; then
            PROFILE_SIZE=$(kubectl exec -n $NAMESPACE $POD_NAME -- ls -l $PROFILE_PATH 2>/dev/null | awk '{print $5}')
            echo "Profile found: $PROFILE_SIZE bytes"
            # Wait a bit more for file to be fully written
            sleep 10
            return 0
        fi

        # Check if pod is still running
        POD_STATUS=$(kubectl get pod $POD_NAME -n $NAMESPACE -o jsonpath='{.status.phase}' 2>/dev/null)
        if [ "$POD_STATUS" = "Failed" ]; then
            print_error "Pod failed"
            return 1
        fi

        echo "  Still generating profile... ($i)"
        sleep 10
    done

    print_error "Timeout waiting for profile"
    return 1
}

profile_server() {
    local SERVER=$1

    print_header "Profiling $SERVER Server"

    local DEPLOYMENT=${DEPLOYMENT_NAMES[$SERVER]}
    local JOB_NAME=${JOB_NAMES[$SERVER]}
    local JOB_FILE=${JOB_FILES[$SERVER]}
    local PROFILE_PATH=${PROFILE_PATHS[$SERVER]}
    local OUTPUT_FILE="$OUTPUT_DIR/${SERVER}_gpu_timeline.nsys-rep"

    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    # Step 1: Check if deployment exists and get replica count
    print_step "Step 1: Checking existing deployment..."
    ORIGINAL_REPLICAS=$(kubectl get deployment $DEPLOYMENT -n $NAMESPACE -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "0")
    echo "Current replicas: $ORIGINAL_REPLICAS"

    # Step 2: Scale down deployment
    if [ "$ORIGINAL_REPLICAS" != "0" ]; then
        print_step "Step 2: Scaling down $DEPLOYMENT..."
        kubectl scale deployment $DEPLOYMENT -n $NAMESPACE --replicas=0
        sleep 10
    else
        echo "Deployment already scaled down or doesn't exist"
    fi

    # Step 3: Clean up any existing profiler job
    print_step "Step 3: Cleaning up existing profiler job..."
    kubectl delete job $JOB_NAME -n $NAMESPACE --force --grace-period=0 2>/dev/null || true
    sleep 5

    # Step 4: Deploy profiler job
    print_step "Step 4: Deploying profiler job..."
    kubectl apply -f "$JOBS_DIR/$JOB_FILE"

    # Step 5: Wait for pod to be ready
    wait_for_pod_ready $JOB_NAME 600

    # Get pod name
    POD_NAME=$(kubectl get pods -n $NAMESPACE -l job-name=$JOB_NAME -o jsonpath='{.items[0].metadata.name}')
    echo "Profiler pod: $POD_NAME"

    # Step 6: Stream logs in background
    print_step "Step 6: Streaming logs (Ctrl+C to skip)..."
    timeout 600 kubectl logs -f $POD_NAME -n $NAMESPACE 2>/dev/null &
    LOGS_PID=$!

    # Step 7: Wait for profile
    wait_for_profile $POD_NAME $PROFILE_PATH 900

    # Kill log streaming
    kill $LOGS_PID 2>/dev/null || true

    # Step 8: Extract profile
    print_step "Step 8: Extracting profile to $OUTPUT_FILE..."
    kubectl cp "$NAMESPACE/$POD_NAME:$PROFILE_PATH" "$OUTPUT_FILE"

    # Verify extraction
    if [ -f "$OUTPUT_FILE" ]; then
        LOCAL_SIZE=$(ls -lh "$OUTPUT_FILE" | awk '{print $5}')
        print_success "Profile extracted: $OUTPUT_FILE ($LOCAL_SIZE)"
    else
        print_error "Failed to extract profile"
    fi

    # Step 9: Cleanup profiler job
    print_step "Step 9: Cleaning up profiler job..."
    kubectl delete job $JOB_NAME -n $NAMESPACE --force --grace-period=0 2>/dev/null || true

    # Step 10: Restore original deployment
    if [ "$ORIGINAL_REPLICAS" != "0" ]; then
        print_step "Step 10: Restoring $DEPLOYMENT to $ORIGINAL_REPLICAS replicas..."
        kubectl scale deployment $DEPLOYMENT -n $NAMESPACE --replicas=$ORIGINAL_REPLICAS
    fi

    print_success "$SERVER profiling complete!"
    echo ""
}

# Main
print_header "LLM Inference Server GPU Profiler"

echo "Server: $SERVER"
echo "Output: $OUTPUT_DIR"
echo "Namespace: $NAMESPACE"
echo ""

# Validate server
if [ "$SERVER" = "all" ]; then
    SERVERS="vllm triton tgi"
elif [ -z "${JOB_NAMES[$SERVER]}" ]; then
    print_error "Unknown server: $SERVER"
    echo "Valid servers: vllm, triton, tgi, all"
    exit 1
else
    SERVERS="$SERVER"
fi

# Run profiling
for S in $SERVERS; do
    profile_server $S
done

print_header "Profiling Complete"

echo "Generated profiles:"
ls -lh "$OUTPUT_DIR"/*.nsys-rep 2>/dev/null || echo "No profiles found"
echo ""
echo "To view profiles:"
echo "  nsys-ui $OUTPUT_DIR/<server>_gpu_timeline.nsys-rep"
