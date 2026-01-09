#!/bin/bash
#
# Extract Nsight Systems Profiles from Kubernetes Pods
#
# This script extracts .nsys-rep profile files from profiler pods.
#
# Usage:
#   ./extract-profiles.sh [output-dir]
#
# Example:
#   ./extract-profiles.sh ./profiles

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

OUTPUT_DIR="${1:-./profiles}"
NAMESPACE="bench"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=============================================="
echo "  Extract Nsight Systems Profiles"
echo "=============================================="
echo ""
echo "Output directory: $OUTPUT_DIR"
echo "Namespace: $NAMESPACE"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Find all profiler pods
echo "Looking for profiler pods..."
PODS=$(kubectl get pods -n $NAMESPACE -l profiler=nsys -o jsonpath='{.items[*].metadata.name}' 2>/dev/null)

if [ -z "$PODS" ]; then
    echo "No profiler pods found with label profiler=nsys"
    echo ""
    echo "Looking for any running profiler jobs..."
    PODS=$(kubectl get pods -n $NAMESPACE | grep -E 'nsys-profiler|nsys-profile' | awk '{print $1}')
fi

if [ -z "$PODS" ]; then
    echo -e "${YELLOW}No profiler pods found${NC}"
    exit 1
fi

echo "Found pods: $PODS"
echo ""

# Extract profiles from each pod
for POD in $PODS; do
    echo "Processing pod: $POD"

    # Check pod status
    STATUS=$(kubectl get pod $POD -n $NAMESPACE -o jsonpath='{.status.phase}' 2>/dev/null)
    echo "  Status: $STATUS"

    if [ "$STATUS" != "Running" ] && [ "$STATUS" != "Succeeded" ]; then
        echo -e "  ${YELLOW}Skipping - pod not ready${NC}"
        continue
    fi

    # Find profile files
    echo "  Looking for .nsys-rep files..."
    PROFILES=$(kubectl exec -n $NAMESPACE $POD -- find /results -name "*.nsys-rep" 2>/dev/null || true)

    if [ -z "$PROFILES" ]; then
        echo -e "  ${YELLOW}No profile files found in /results${NC}"
        continue
    fi

    # Extract each profile
    for PROFILE in $PROFILES; do
        FILENAME=$(basename $PROFILE)
        LOCAL_FILE="$OUTPUT_DIR/${FILENAME%.nsys-rep}_${TIMESTAMP}.nsys-rep"

        echo "  Extracting: $PROFILE"
        kubectl cp "$NAMESPACE/$POD:$PROFILE" "$LOCAL_FILE"

        if [ -f "$LOCAL_FILE" ]; then
            SIZE=$(ls -lh "$LOCAL_FILE" | awk '{print $5}')
            echo -e "  ${GREEN}Saved: $LOCAL_FILE ($SIZE)${NC}"
        else
            echo -e "  ${RED}Failed to extract${NC}"
        fi
    done

    echo ""
done

echo "=============================================="
echo "  Extraction Complete"
echo "=============================================="
echo ""
echo "Extracted profiles:"
ls -lh "$OUTPUT_DIR"/*.nsys-rep 2>/dev/null || echo "No profiles extracted"
echo ""
echo "To view profiles:"
echo "  nsys-ui <profile>.nsys-rep"
