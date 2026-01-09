#!/usr/bin/env bash
# Collect results from benchmark pods
# Usage: ./collect_results.sh [job-name]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="${PROJECT_ROOT}/results"
PROFILES_DIR="${RESULTS_DIR}/profiles"

mkdir -p "${PROFILES_DIR}"

echo "======================================================="
echo "  Collecting Results"
echo "======================================================="
echo ""

# Find profiler pods
echo "Looking for profiler pods..."
PROFILER_PODS=$(kubectl get pods -n bench -l job-name -o jsonpath='{.items[*].metadata.name}' 2>/dev/null | tr ' ' '\n' | grep -E "profiler-" || true)

if [ -z "$PROFILER_PODS" ]; then
    echo "No profiler pods found."
else
    echo "Found profiler pods:"
    echo "$PROFILER_PODS"
    echo ""

    for pod in $PROFILER_PODS; do
        echo "Copying from ${pod}..."

        # Copy profiles
        kubectl cp bench/${pod}:/results/profiles/ "${PROFILES_DIR}/" 2>/dev/null || \
            echo "  Warning: Could not copy profiles from ${pod}"

        # Copy benchmark results
        kubectl cp bench/${pod}:/results/benchmarks/ "${RESULTS_DIR}/benchmarks/" 2>/dev/null || \
            echo "  Warning: Could not copy benchmarks from ${pod}"
    done
fi

echo ""
echo "======================================================="
echo "  Results Collection Complete"
echo "======================================================="
echo ""
echo "Profiles directory: ${PROFILES_DIR}"
ls -la "${PROFILES_DIR}"/ 2>/dev/null || echo "  (empty)"

echo ""
echo "Benchmarks directory: ${RESULTS_DIR}/benchmarks"
ls -la "${RESULTS_DIR}/benchmarks"/ 2>/dev/null || echo "  (empty)"
echo ""
