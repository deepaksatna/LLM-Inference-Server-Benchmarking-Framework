#!/usr/bin/env python3
"""
LLM Inference Benchmark Orchestrator
Runs comprehensive benchmark matrix across all backends.

Usage:
  python3 benchmark.py --backends vllm triton tgi
  python3 benchmark.py --matrix configs/benchmark_matrix.yaml
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import yaml

# Benchmark matrix configuration
DEFAULT_MATRIX = {
    "backends": ["vllm", "triton", "tgi"],
    "concurrency": [1, 8, 32],
    "max_tokens": [128, 512],
    "iterations": 100,
    "warmup": 10,
}

# OCI GPU hourly rates (USD)
GPU_HOURLY_RATES = {
    "A10": 1.50,
    "A100": 4.00,
    "H100": 8.00,
}


def run_single_benchmark(
    backend: str,
    iterations: int,
    warmup: int,
    max_tokens: int,
    concurrency: int,
    output_dir: Path,
) -> Optional[Dict]:
    """Run a single benchmark configuration."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{backend}_c{concurrency}_t{max_tokens}_{timestamp}.json"

    print(f"\n{'='*60}")
    print(f"  Running: {backend} | concurrency={concurrency} | tokens={max_tokens}")
    print(f"{'='*60}")

    cmd = [
        "python3", "inference_client.py",
        "--backend", backend,
        "--iterations", str(iterations),
        "--warmup", str(warmup),
        "--max-tokens", str(max_tokens),
        "--concurrency", str(concurrency),
        "--output-file", str(output_file),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        if result.returncode != 0:
            print(f"ERROR: Benchmark failed")
            print(result.stderr[:500] if result.stderr else "No error output")
            return None

        # Load results
        with open(output_file) as f:
            data = json.load(f)

        return data

    except subprocess.TimeoutExpired:
        print(f"ERROR: Benchmark timed out")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def calculate_cost(
    tokens_per_sec: float,
    gpu_type: str = "A10",
    tokens_to_generate: int = 1_000_000,
) -> float:
    """Calculate cost per million tokens."""
    hourly_rate = GPU_HOURLY_RATES.get(gpu_type, 1.50)

    # Time to generate 1M tokens
    hours = tokens_to_generate / (tokens_per_sec * 3600)

    # Cost
    cost = hours * hourly_rate

    return cost


def run_benchmark_matrix(
    matrix: Dict,
    output_dir: Path,
    gpu_type: str = "A10",
) -> Dict:
    """Run full benchmark matrix."""

    all_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "gpu_type": gpu_type,
            "gpu_hourly_rate": GPU_HOURLY_RATES.get(gpu_type, 1.50),
            "matrix": matrix,
        },
        "results": [],
    }

    total_tests = len(matrix["backends"]) * len(matrix["concurrency"]) * len(matrix["max_tokens"])
    current = 0

    print(f"\n{'='*70}")
    print(f"  LLM Inference Benchmark Matrix")
    print(f"{'='*70}")
    print(f"  Backends:    {matrix['backends']}")
    print(f"  Concurrency: {matrix['concurrency']}")
    print(f"  Max Tokens:  {matrix['max_tokens']}")
    print(f"  Iterations:  {matrix['iterations']}")
    print(f"  Total Tests: {total_tests}")
    print(f"{'='*70}")

    for backend in matrix["backends"]:
        for concurrency in matrix["concurrency"]:
            for max_tokens in matrix["max_tokens"]:
                current += 1
                print(f"\n[{current}/{total_tests}]", end="")

                result = run_single_benchmark(
                    backend=backend,
                    iterations=matrix["iterations"],
                    warmup=matrix["warmup"],
                    max_tokens=max_tokens,
                    concurrency=concurrency,
                    output_dir=output_dir,
                )

                if result:
                    # Add cost calculation
                    tokens_per_sec = result["statistics"]["throughput"]["tokens_per_sec"]
                    cost_per_million = calculate_cost(tokens_per_sec, gpu_type)

                    result["cost"] = {
                        "gpu_type": gpu_type,
                        "hourly_rate": GPU_HOURLY_RATES.get(gpu_type, 1.50),
                        "per_million_tokens": cost_per_million,
                    }

                    all_results["results"].append(result)

                    print(f"  Throughput: {tokens_per_sec:.1f} tok/s | Cost: ${cost_per_million:.4f}/1M tokens")
                else:
                    all_results["results"].append({
                        "backend": backend,
                        "config": {
                            "concurrency": concurrency,
                            "max_tokens": max_tokens,
                        },
                        "status": "failed",
                    })

    return all_results


def generate_summary(all_results: Dict) -> str:
    """Generate text summary of benchmark results."""
    lines = []
    lines.append("=" * 70)
    lines.append("  LLM INFERENCE BENCHMARK SUMMARY")
    lines.append("=" * 70)
    lines.append(f"  Generated: {all_results['metadata']['timestamp']}")
    lines.append(f"  GPU Type:  {all_results['metadata']['gpu_type']}")
    lines.append("")

    # Group by backend
    by_backend = {}
    for result in all_results["results"]:
        backend = result["backend"]
        if backend not in by_backend:
            by_backend[backend] = []
        by_backend[backend].append(result)

    # Best results per backend
    lines.append("-" * 70)
    lines.append("  BEST RESULTS PER BACKEND")
    lines.append("-" * 70)

    for backend, results in by_backend.items():
        successful = [r for r in results if r.get("status") != "failed" and "statistics" in r]

        if not successful:
            lines.append(f"\n{backend.upper()}: No successful runs")
            continue

        # Best throughput
        best_throughput = max(successful, key=lambda x: x["statistics"]["throughput"]["tokens_per_sec"])
        # Lowest latency (at concurrency=1)
        c1_results = [r for r in successful if r["config"]["concurrency"] == 1]
        best_latency = min(c1_results, key=lambda x: x["statistics"]["latency_ms"]["p50"]) if c1_results else None
        # Lowest cost
        best_cost = min(successful, key=lambda x: x.get("cost", {}).get("per_million_tokens", float("inf")))

        lines.append(f"\n{backend.upper()}:")
        lines.append(f"  Best Throughput: {best_throughput['statistics']['throughput']['tokens_per_sec']:.1f} tok/s")
        lines.append(f"    (concurrency={best_throughput['config']['concurrency']}, tokens={best_throughput['config']['max_tokens']})")

        if best_latency:
            lines.append(f"  Best Latency (P50): {best_latency['statistics']['latency_ms']['p50']:.1f} ms")
            lines.append(f"    (concurrency=1, tokens={best_latency['config']['max_tokens']})")

        if best_cost.get("cost"):
            lines.append(f"  Best Cost: ${best_cost['cost']['per_million_tokens']:.4f}/1M tokens")

    # Comparison table
    lines.append("")
    lines.append("-" * 70)
    lines.append("  DETAILED COMPARISON")
    lines.append("-" * 70)
    lines.append("")
    lines.append(f"{'Backend':<10} {'Concur':<8} {'Tokens':<8} {'Throughput':<12} {'P50(ms)':<10} {'P95(ms)':<10} {'$/1M':<10}")
    lines.append("-" * 70)

    for result in all_results["results"]:
        if result.get("status") == "failed" or "statistics" not in result:
            continue

        stats = result["statistics"]
        cost = result.get("cost", {}).get("per_million_tokens", 0)

        lines.append(
            f"{result['backend']:<10} "
            f"{result['config']['concurrency']:<8} "
            f"{result['config']['max_tokens']:<8} "
            f"{stats['throughput']['tokens_per_sec']:<12.1f} "
            f"{stats['latency_ms']['p50']:<10.1f} "
            f"{stats['latency_ms']['p95']:<10.1f} "
            f"${cost:<9.4f}"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="LLM Inference Benchmark Orchestrator")
    parser.add_argument("--backends", nargs="+", default=["vllm", "triton", "tgi"],
                       help="Backends to benchmark")
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 8, 32],
                       help="Concurrency levels to test")
    parser.add_argument("--max-tokens", nargs="+", type=int, default=[128, 512],
                       help="Max tokens values to test")
    parser.add_argument("--iterations", type=int, default=100,
                       help="Iterations per test")
    parser.add_argument("--warmup", type=int, default=10,
                       help="Warmup iterations")
    parser.add_argument("--matrix", type=str, default=None,
                       help="YAML file with benchmark matrix")
    parser.add_argument("--gpu-type", type=str, default="A10",
                       choices=["A10", "A100", "H100"],
                       help="GPU type for cost calculation")
    parser.add_argument("--output-dir", type=str, default="./results/benchmarks",
                       help="Output directory")

    args = parser.parse_args()

    # Load or create matrix
    if args.matrix:
        with open(args.matrix) as f:
            matrix = yaml.safe_load(f)
    else:
        matrix = {
            "backends": args.backends,
            "concurrency": args.concurrency,
            "max_tokens": args.max_tokens,
            "iterations": args.iterations,
            "warmup": args.warmup,
        }

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run benchmarks
    all_results = run_benchmark_matrix(matrix, output_dir, args.gpu_type)

    # Save combined results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_file = output_dir / f"all_results_{timestamp}.json"
    with open(combined_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {combined_file}")

    # Generate and save summary
    summary = generate_summary(all_results)
    summary_file = output_dir / f"summary_{timestamp}.txt"
    with open(summary_file, "w") as f:
        f.write(summary)

    print(f"Summary saved to: {summary_file}")
    print()
    print(summary)

    return 0


if __name__ == "__main__":
    sys.exit(main())
