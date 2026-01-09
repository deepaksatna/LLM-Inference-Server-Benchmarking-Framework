#!/usr/bin/env python3
"""
LLM Inference Benchmark Visualization
Generates comparison plots using matplotlib only (no seaborn).

Usage:
  python3 visualize.py --input results/benchmarks/all_results_*.json
  python3 visualize.py --input-dir results/benchmarks/
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

# Backend configuration
BACKENDS = {
    "vllm": {"color": "#2ecc71", "label": "vLLM", "marker": "o"},
    "triton": {"color": "#3498db", "label": "NVIDIA Triton", "marker": "s"},
    "tgi": {"color": "#e74c3c", "label": "HuggingFace TGI", "marker": "^"},
}


def load_results(input_path: Path) -> Dict:
    """Load benchmark results from JSON file."""
    with open(input_path) as f:
        return json.load(f)


def group_by_backend(results: List[Dict]) -> Dict[str, List[Dict]]:
    """Group results by backend."""
    grouped = {}
    for result in results:
        if result.get("status") == "failed" or "statistics" not in result:
            continue
        backend = result["backend"]
        if backend not in grouped:
            grouped[backend] = []
        grouped[backend].append(result)
    return grouped


def create_throughput_plot(results: List[Dict], output_dir: Path):
    """Create throughput vs concurrency plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    grouped = group_by_backend(results)

    for backend, data in grouped.items():
        config = BACKENDS.get(backend, {"color": "gray", "label": backend, "marker": "o"})

        # Group by concurrency
        by_concurrency = {}
        for r in data:
            c = r["config"]["concurrency"]
            if c not in by_concurrency:
                by_concurrency[c] = []
            by_concurrency[c].append(r["statistics"]["throughput"]["tokens_per_sec"])

        concurrencies = sorted(by_concurrency.keys())
        throughputs = [np.mean(by_concurrency[c]) for c in concurrencies]

        ax.plot(
            concurrencies,
            throughputs,
            marker=config["marker"],
            color=config["color"],
            label=config["label"],
            linewidth=2,
            markersize=8,
        )

    ax.set_xlabel("Concurrency", fontsize=12, fontweight="bold")
    ax.set_ylabel("Throughput (tokens/sec)", fontsize=12, fontweight="bold")
    ax.set_title("LLM Inference Throughput vs Concurrency\nHigher is Better", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)

    plt.tight_layout()
    output_file = output_dir / "throughput_comparison.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Created: {output_file}")


def create_latency_plot(results: List[Dict], output_dir: Path):
    """Create P50 latency vs concurrency plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    grouped = group_by_backend(results)

    for backend, data in grouped.items():
        config = BACKENDS.get(backend, {"color": "gray", "label": backend, "marker": "o"})

        # Group by concurrency
        by_concurrency = {}
        for r in data:
            c = r["config"]["concurrency"]
            if c not in by_concurrency:
                by_concurrency[c] = []
            by_concurrency[c].append(r["statistics"]["latency_ms"]["p50"])

        concurrencies = sorted(by_concurrency.keys())
        latencies = [np.mean(by_concurrency[c]) for c in concurrencies]

        ax.plot(
            concurrencies,
            latencies,
            marker=config["marker"],
            color=config["color"],
            label=config["label"],
            linewidth=2,
            markersize=8,
        )

    ax.set_xlabel("Concurrency", fontsize=12, fontweight="bold")
    ax.set_ylabel("P50 Latency (ms)", fontsize=12, fontweight="bold")
    ax.set_title("LLM Inference Latency (P50) vs Concurrency\nLower is Better", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)

    plt.tight_layout()
    output_file = output_dir / "latency_p50_comparison.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Created: {output_file}")


def create_latency_percentiles_plot(results: List[Dict], output_dir: Path):
    """Create latency percentiles bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6))

    grouped = group_by_backend(results)
    percentiles = ["p50", "p90", "p95", "p99"]

    # Get data for concurrency=1 (baseline latency)
    c1_data = {}
    for backend, data in grouped.items():
        c1 = [r for r in data if r["config"]["concurrency"] == 1]
        if c1:
            c1_data[backend] = c1[0]["statistics"]["latency_ms"]

    if not c1_data:
        print("  Warning: No concurrency=1 data found for percentiles plot")
        return

    x = np.arange(len(percentiles))
    width = 0.25
    multiplier = 0

    for backend, latencies in c1_data.items():
        config = BACKENDS.get(backend, {"color": "gray", "label": backend})
        values = [latencies[p] for p in percentiles]

        offset = width * multiplier
        bars = ax.bar(x + offset, values, width, label=config["label"], color=config["color"])

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        multiplier += 1

    ax.set_xlabel("Percentile", fontsize=12, fontweight="bold")
    ax.set_ylabel("Latency (ms)", fontsize=12, fontweight="bold")
    ax.set_title("Latency Percentiles (Concurrency=1)\nLower is Better", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels([p.upper() for p in percentiles])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_file = output_dir / "latency_percentiles.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Created: {output_file}")


def create_cost_comparison_plot(results: List[Dict], output_dir: Path):
    """Create cost per million tokens bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    grouped = group_by_backend(results)

    # Get best cost for each backend
    best_costs = {}
    for backend, data in grouped.items():
        costs = [r.get("cost", {}).get("per_million_tokens", float("inf")) for r in data]
        costs = [c for c in costs if c < float("inf")]
        if costs:
            best_costs[backend] = min(costs)

    if not best_costs:
        print("  Warning: No cost data found")
        return

    backends = list(best_costs.keys())
    costs = [best_costs[b] for b in backends]
    colors = [BACKENDS.get(b, {}).get("color", "gray") for b in backends]
    labels = [BACKENDS.get(b, {}).get("label", b) for b in backends]

    bars = ax.bar(labels, costs, color=colors, edgecolor="black", linewidth=1.2)

    # Add value labels
    for bar, cost in zip(bars, costs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"${cost:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xlabel("Backend", fontsize=12, fontweight="bold")
    ax.set_ylabel("Cost per Million Tokens ($)", fontsize=12, fontweight="bold")
    ax.set_title("Cost Comparison (Best Configuration)\nLower is Better", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_file = output_dir / "cost_comparison.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Created: {output_file}")


def create_summary_dashboard(results: List[Dict], output_dir: Path):
    """Create multi-panel summary dashboard."""
    fig = plt.figure(figsize=(14, 10))

    grouped = group_by_backend(results)

    # Panel 1: Throughput
    ax1 = fig.add_subplot(2, 2, 1)
    for backend, data in grouped.items():
        config = BACKENDS.get(backend, {"color": "gray", "label": backend, "marker": "o"})
        by_c = {}
        for r in data:
            c = r["config"]["concurrency"]
            if c not in by_c:
                by_c[c] = []
            by_c[c].append(r["statistics"]["throughput"]["tokens_per_sec"])
        cs = sorted(by_c.keys())
        ts = [np.mean(by_c[c]) for c in cs]
        ax1.plot(cs, ts, marker=config["marker"], color=config["color"], label=config["label"], linewidth=2)
    ax1.set_xlabel("Concurrency", fontweight="bold")
    ax1.set_ylabel("Tokens/sec", fontweight="bold")
    ax1.set_title("Throughput", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log", base=2)

    # Panel 2: Latency P50
    ax2 = fig.add_subplot(2, 2, 2)
    for backend, data in grouped.items():
        config = BACKENDS.get(backend, {"color": "gray", "label": backend, "marker": "o"})
        by_c = {}
        for r in data:
            c = r["config"]["concurrency"]
            if c not in by_c:
                by_c[c] = []
            by_c[c].append(r["statistics"]["latency_ms"]["p50"])
        cs = sorted(by_c.keys())
        ls = [np.mean(by_c[c]) for c in cs]
        ax2.plot(cs, ls, marker=config["marker"], color=config["color"], label=config["label"], linewidth=2)
    ax2.set_xlabel("Concurrency", fontweight="bold")
    ax2.set_ylabel("P50 Latency (ms)", fontweight="bold")
    ax2.set_title("Latency (P50)", fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale("log", base=2)

    # Panel 3: Latency P95
    ax3 = fig.add_subplot(2, 2, 3)
    for backend, data in grouped.items():
        config = BACKENDS.get(backend, {"color": "gray", "label": backend, "marker": "o"})
        by_c = {}
        for r in data:
            c = r["config"]["concurrency"]
            if c not in by_c:
                by_c[c] = []
            by_c[c].append(r["statistics"]["latency_ms"]["p95"])
        cs = sorted(by_c.keys())
        ls = [np.mean(by_c[c]) for c in cs]
        ax3.plot(cs, ls, marker=config["marker"], color=config["color"], label=config["label"], linewidth=2)
    ax3.set_xlabel("Concurrency", fontweight="bold")
    ax3.set_ylabel("P95 Latency (ms)", fontweight="bold")
    ax3.set_title("Latency (P95)", fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale("log", base=2)

    # Panel 4: Cost
    ax4 = fig.add_subplot(2, 2, 4)
    best_costs = {}
    for backend, data in grouped.items():
        costs = [r.get("cost", {}).get("per_million_tokens", float("inf")) for r in data]
        costs = [c for c in costs if c < float("inf")]
        if costs:
            best_costs[backend] = min(costs)

    if best_costs:
        backends = list(best_costs.keys())
        costs = [best_costs[b] for b in backends]
        colors = [BACKENDS.get(b, {}).get("color", "gray") for b in backends]
        labels = [BACKENDS.get(b, {}).get("label", b) for b in backends]
        bars = ax4.bar(labels, costs, color=colors)
        for bar, cost in zip(bars, costs):
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"${cost:.4f}",
                    ha="center", va="bottom", fontsize=9)
    ax4.set_ylabel("$/1M Tokens", fontweight="bold")
    ax4.set_title("Cost (Best Config)", fontweight="bold")
    ax4.grid(True, alpha=0.3, axis="y")

    fig.suptitle("LLM Inference Benchmark Summary", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()

    output_file = output_dir / "summary_dashboard.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Created: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="LLM Inference Benchmark Visualization")
    parser.add_argument("--input", type=str, help="Input JSON file with all results")
    parser.add_argument("--input-dir", type=str, help="Directory with benchmark results")
    parser.add_argument("--output-dir", type=str, default="./results/plots",
                       help="Output directory for plots")

    args = parser.parse_args()

    # Find input file
    if args.input:
        input_files = [Path(args.input)]
    elif args.input_dir:
        input_files = sorted(Path(args.input_dir).glob("all_results_*.json"))
    else:
        input_files = sorted(Path("./results/benchmarks").glob("all_results_*.json"))

    if not input_files:
        print("ERROR: No benchmark results found")
        print("Run: python3 benchmark.py first")
        return 1

    # Use most recent file
    input_file = input_files[-1]
    print(f"Loading results from: {input_file}")

    data = load_results(input_file)
    results = data.get("results", [])

    if not results:
        print("ERROR: No results found in file")
        return 1

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating visualizations...")

    # Generate plots
    create_throughput_plot(results, output_dir)
    create_latency_plot(results, output_dir)
    create_latency_percentiles_plot(results, output_dir)
    create_cost_comparison_plot(results, output_dir)
    create_summary_dashboard(results, output_dir)

    print(f"\nAll plots saved to: {output_dir}")
    print(f"Files:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  - {f.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
