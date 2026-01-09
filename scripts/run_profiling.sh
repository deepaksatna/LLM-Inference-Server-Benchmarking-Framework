#!/usr/bin/env bash
# Run Nsight Systems profiling on inference backend
# Usage: ./run_profiling.sh <backend> [iterations]
#   backend: vllm, triton, tgi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

BACKEND="${1:-vllm}"
ITERATIONS="${2:-50}"
WARMUP="${3:-10}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "======================================================="
echo "  Nsight Systems Profiling: ${BACKEND}"
echo "======================================================="
echo ""
echo "Configuration:"
echo "  Backend:    ${BACKEND}"
echo "  Iterations: ${ITERATIONS}"
echo "  Warmup:     ${WARMUP}"
echo ""

# Verify backend is running
DEPLOYMENT_NAME="${BACKEND}-server"
POD_NAME=$(kubectl get pods -n bench -l app=${DEPLOYMENT_NAME} -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

if [ -z "$POD_NAME" ]; then
    echo "ERROR: No pods found for ${BACKEND}"
    echo "Deploy first: ./scripts/deploy_backend.sh ${BACKEND} --wait"
    exit 1
fi

echo "Target pod: ${POD_NAME}"
echo ""

# Create profiler job
PROFILER_JOB="profiler-${BACKEND}-${TIMESTAMP}"

cat << EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: ${PROFILER_JOB}
  namespace: bench
spec:
  backoffLimit: 1
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: profiler
        image: fra.ocir.io/frntrd2vyxvi/models:llm-bench-client-v1
        command: ["/bin/bash", "-c"]
        args:
        - |
          set -e
          echo "Starting profiler..."

          # Wait for backend to be ready
          ENDPOINT=""
          case "${BACKEND}" in
            vllm)   ENDPOINT="http://vllm-service:8000/health" ;;
            triton) ENDPOINT="http://triton-service:8000/v2/health/ready" ;;
            tgi)    ENDPOINT="http://tgi-service:8000/health" ;;
          esac

          echo "Waiting for backend at \${ENDPOINT}..."
          for i in \$(seq 1 60); do
            if curl -s "\${ENDPOINT}" > /dev/null 2>&1; then
              echo "Backend ready!"
              break
            fi
            echo "  Waiting... (\$i/60)"
            sleep 5
          done

          # Run profiling
          cd /app
          ./nsys/scripts/profile_inference.sh ${BACKEND} ${ITERATIONS} ${WARMUP}

          echo ""
          echo "Profiling complete. Keeping pod alive for file copy..."
          echo "Copy profiles with:"
          echo "  kubectl cp bench/${PROFILER_JOB}-xxx:/results/profiles/ ./results/profiles/"
          sleep 300

        env:
        - name: BACKEND
          value: "${BACKEND}"
        - name: NSYS_OUTPUT_DIR
          value: "/results/profiles"

        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: "1"
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: "1"

        volumeMounts:
        - name: results
          mountPath: /results
        - name: dshm
          mountPath: /dev/shm

        securityContext:
          capabilities:
            add: ["SYS_ADMIN", "SYS_PTRACE"]

      volumes:
      - name: results
        emptyDir: {}
      - name: dshm
        emptyDir:
          medium: Memory
          sizeLimit: "4Gi"

      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule

      nodeSelector:
        nvidia.com/gpu: "true"
EOF

echo ""
echo "Profiler job created: ${PROFILER_JOB}"
echo ""
echo "Monitor with:"
echo "  kubectl logs -f job/${PROFILER_JOB} -n bench"
echo ""
echo "When complete, copy profiles:"
echo "  POD=\$(kubectl get pods -n bench -l job-name=${PROFILER_JOB} -o jsonpath='{.items[0].metadata.name}')"
echo "  kubectl cp bench/\${POD}:/results/profiles/ ./results/profiles/"
echo ""
