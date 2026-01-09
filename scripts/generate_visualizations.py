#!/usr/bin/env python3
"""
LLM Inference Benchmark Visualization Script
Generates comprehensive performance comparison graphs for vLLM, Triton, and TGI
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Color scheme
COLORS = {
    'vllm': '#2ecc71',      # Green
    'triton': '#3498db',    # Blue
    'tgi': '#e74c3c'        # Red
}

BACKEND_NAMES = {
    'vllm': 'vLLM',
    'triton': 'NVIDIA Triton',
    'tgi': 'HuggingFace TGI'
}

def load_benchmark_data(benchmark_dir):
    """Load all benchmark JSON files"""
    results = []
    for f in Path(benchmark_dir).glob("*.json"):
        if f.name.startswith(('vllm_', 'triton_', 'tgi_')):
            with open(f) as fp:
                data = json.load(fp)
                results.append(data)
    return pd.DataFrame(results)

def load_gpu_metrics(profile_dir):
    """Load nvidia-smi dmon logs"""
    metrics = {}
    for backend in ['vllm', 'triton', 'tgi']:
        log_file = Path(profile_dir) / f"{backend}_gpu_dmon.log"
        if log_file.exists():
            data = []
            with open(log_file) as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    parts = line.split()
                    if len(parts) >= 6:
                        try:
                            data.append({
                                'gpu': int(parts[0]),
                                'power': int(parts[1]),
                                'temp': int(parts[2]),
                                'sm': int(parts[4]),
                                'mem': int(parts[5])
                            })
                        except (ValueError, IndexError):
                            continue
            metrics[backend] = pd.DataFrame(data)
    return metrics

def plot_throughput_comparison(df, output_dir):
    """Generate throughput comparison charts"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Throughput by concurrency (50 tokens)
    ax1 = axes[0]
    df_50 = df[df['max_tokens'] == 50]

    for backend in ['vllm', 'triton', 'tgi']:
        backend_data = df_50[df_50['backend'] == backend].sort_values('concurrency')
        ax1.plot(backend_data['concurrency'], backend_data['throughput_rps'],
                marker='o', linewidth=2, markersize=8,
                color=COLORS[backend], label=BACKEND_NAMES[backend])

    ax1.set_xlabel('Concurrency')
    ax1.set_ylabel('Throughput (requests/sec)')
    ax1.set_title('Throughput vs Concurrency (50 tokens)')
    ax1.legend()
    ax1.set_xticks([1, 4, 8, 16])
    ax1.grid(True, alpha=0.3)

    # Throughput by concurrency (100 tokens)
    ax2 = axes[1]
    df_100 = df[df['max_tokens'] == 100]

    for backend in ['vllm', 'triton', 'tgi']:
        backend_data = df_100[df_100['backend'] == backend].sort_values('concurrency')
        ax2.plot(backend_data['concurrency'], backend_data['throughput_rps'],
                marker='s', linewidth=2, markersize=8,
                color=COLORS[backend], label=BACKEND_NAMES[backend])

    ax2.set_xlabel('Concurrency')
    ax2.set_ylabel('Throughput (requests/sec)')
    ax2.set_title('Throughput vs Concurrency (100 tokens)')
    ax2.legend()
    ax2.set_xticks([1, 4, 8, 16])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'throughput_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: throughput_comparison.png")

def plot_tokens_per_second(df, output_dir):
    """Generate tokens/second comparison"""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(4)  # concurrency levels
    width = 0.25
    concurrencies = [1, 4, 8, 16]

    df_100 = df[df['max_tokens'] == 100]

    for i, backend in enumerate(['vllm', 'triton', 'tgi']):
        backend_data = df_100[df_100['backend'] == backend].sort_values('concurrency')
        values = backend_data['tokens_per_sec'].values
        ax.bar(x + i*width, values, width, label=BACKEND_NAMES[backend], color=COLORS[backend])

    ax.set_xlabel('Concurrency Level')
    ax.set_ylabel('Tokens per Second')
    ax.set_title('Token Generation Rate Comparison (100 tokens output)')
    ax.set_xticks(x + width)
    ax.set_xticklabels(concurrencies)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, backend in enumerate(['vllm', 'triton', 'tgi']):
        backend_data = df_100[df_100['backend'] == backend].sort_values('concurrency')
        for j, val in enumerate(backend_data['tokens_per_sec'].values):
            ax.annotate(f'{val:.0f}', xy=(j + i*width, val), ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'tokens_per_second.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: tokens_per_second.png")

def plot_latency_comparison(df, output_dir):
    """Generate latency comparison charts"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # P50 Latency
    ax1 = axes[0, 0]
    df_50 = df[df['max_tokens'] == 50]
    for backend in ['vllm', 'triton', 'tgi']:
        backend_data = df_50[df_50['backend'] == backend].sort_values('concurrency')
        ax1.plot(backend_data['concurrency'], backend_data['p50_latency_ms'],
                marker='o', linewidth=2, color=COLORS[backend], label=BACKEND_NAMES[backend])
    ax1.set_xlabel('Concurrency')
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('P50 Latency vs Concurrency (50 tokens)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # P95 Latency
    ax2 = axes[0, 1]
    for backend in ['vllm', 'triton', 'tgi']:
        backend_data = df_50[df_50['backend'] == backend].sort_values('concurrency')
        ax2.plot(backend_data['concurrency'], backend_data['p95_latency_ms'],
                marker='s', linewidth=2, color=COLORS[backend], label=BACKEND_NAMES[backend])
    ax2.set_xlabel('Concurrency')
    ax2.set_ylabel('Latency (ms)')
    ax2.set_title('P95 Latency vs Concurrency (50 tokens)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Latency distribution at c=16
    ax3 = axes[1, 0]
    df_c16 = df[(df['concurrency'] == 16) & (df['max_tokens'] == 50)]
    x = np.arange(3)
    width = 0.2

    metrics = ['avg_latency_ms', 'p50_latency_ms', 'p95_latency_ms', 'p99_latency_ms']
    metric_names = ['Avg', 'P50', 'P95', 'P99']

    for i, backend in enumerate(['vllm', 'triton', 'tgi']):
        backend_row = df_c16[df_c16['backend'] == backend].iloc[0]
        values = [backend_row[m] for m in metrics]
        ax3.bar(np.arange(4) + i*width, values, width, label=BACKEND_NAMES[backend], color=COLORS[backend])

    ax3.set_xlabel('Latency Percentile')
    ax3.set_ylabel('Latency (ms)')
    ax3.set_title('Latency Distribution at Concurrency=16 (50 tokens)')
    ax3.set_xticks(np.arange(4) + width)
    ax3.set_xticklabels(metric_names)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Latency at c=1 (single request)
    ax4 = axes[1, 1]
    df_c1 = df[(df['concurrency'] == 1)]

    for backend in ['vllm', 'triton', 'tgi']:
        backend_data = df_c1[df_c1['backend'] == backend].sort_values('max_tokens')
        ax4.bar([50, 100], backend_data['avg_latency_ms'].values,
               alpha=0.7, label=BACKEND_NAMES[backend], color=COLORS[backend], width=15)

    ax4.set_xlabel('Output Tokens')
    ax4.set_ylabel('Average Latency (ms)')
    ax4.set_title('Single Request Latency by Output Length')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'latency_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: latency_comparison.png")

def plot_gpu_metrics(gpu_metrics, output_dir):
    """Generate GPU metrics comparison"""
    if not gpu_metrics:
        print("No GPU metrics data available")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # SM Utilization over time
    ax1 = axes[0, 0]
    for backend, data in gpu_metrics.items():
        if not data.empty:
            ax1.plot(range(len(data)), data['sm'], label=BACKEND_NAMES[backend],
                    color=COLORS[backend], linewidth=2)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('SM Utilization (%)')
    ax1.set_title('GPU SM Utilization During Inference')
    ax1.legend()
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3)

    # Power consumption over time
    ax2 = axes[0, 1]
    for backend, data in gpu_metrics.items():
        if not data.empty:
            ax2.plot(range(len(data)), data['power'], label=BACKEND_NAMES[backend],
                    color=COLORS[backend], linewidth=2)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Power (Watts)')
    ax2.set_title('GPU Power Consumption During Inference')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Average SM Utilization comparison
    ax3 = axes[1, 0]
    avg_sm = []
    peak_sm = []
    backends_list = []
    for backend in ['vllm', 'triton', 'tgi']:
        if backend in gpu_metrics and not gpu_metrics[backend].empty:
            data = gpu_metrics[backend]
            # Filter to active inference (SM > 50%)
            active = data[data['sm'] > 50]
            if not active.empty:
                avg_sm.append(active['sm'].mean())
                peak_sm.append(active['sm'].max())
                backends_list.append(BACKEND_NAMES[backend])

    x = np.arange(len(backends_list))
    width = 0.35
    ax3.bar(x - width/2, avg_sm, width, label='Average SM%', color='steelblue')
    ax3.bar(x + width/2, peak_sm, width, label='Peak SM%', color='coral')
    ax3.set_ylabel('SM Utilization (%)')
    ax3.set_title('GPU Compute Utilization Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(backends_list)
    ax3.legend()
    ax3.set_ylim(0, 105)
    ax3.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (avg, peak) in enumerate(zip(avg_sm, peak_sm)):
        ax3.annotate(f'{avg:.1f}%', xy=(i - width/2, avg), ha='center', va='bottom', fontsize=10)
        ax3.annotate(f'{peak:.0f}%', xy=(i + width/2, peak), ha='center', va='bottom', fontsize=10)

    # Power and Temperature comparison
    ax4 = axes[1, 1]
    avg_power = []
    peak_power = []
    max_temp = []
    backends_list = []
    for backend in ['vllm', 'triton', 'tgi']:
        if backend in gpu_metrics and not gpu_metrics[backend].empty:
            data = gpu_metrics[backend]
            active = data[data['sm'] > 50]
            if not active.empty:
                avg_power.append(active['power'].mean())
                peak_power.append(active['power'].max())
                max_temp.append(active['temp'].max())
                backends_list.append(BACKEND_NAMES[backend])

    x = np.arange(len(backends_list))
    ax4.bar(x, peak_power, 0.6, label='Peak Power (W)', color=[COLORS[b.lower().replace(' ', '').replace('nvidia', '').replace('huggingface', '')] for b in backends_list])
    ax4.set_ylabel('Power (Watts)')
    ax4.set_title('Peak Power Consumption During Inference')
    ax4.set_xticks(x)
    ax4.set_xticklabels(backends_list)
    ax4.grid(True, alpha=0.3, axis='y')

    # Add temperature as text
    for i, (pwr, tmp) in enumerate(zip(peak_power, max_temp)):
        ax4.annotate(f'{pwr:.0f}W\n{tmp}°C', xy=(i, pwr), ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'gpu_metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: gpu_metrics_comparison.png")

def plot_scaling_efficiency(df, output_dir):
    """Generate scaling efficiency chart"""
    fig, ax = plt.subplots(figsize=(10, 6))

    df_50 = df[df['max_tokens'] == 50]

    for backend in ['vllm', 'triton', 'tgi']:
        backend_data = df_50[df_50['backend'] == backend].sort_values('concurrency')

        # Calculate scaling efficiency (throughput / ideal_throughput * 100)
        base_throughput = backend_data[backend_data['concurrency'] == 1]['throughput_rps'].values[0]
        concurrencies = backend_data['concurrency'].values
        actual_throughput = backend_data['throughput_rps'].values
        ideal_throughput = base_throughput * concurrencies
        efficiency = (actual_throughput / ideal_throughput) * 100

        ax.plot(concurrencies, efficiency, marker='o', linewidth=2, markersize=8,
               color=COLORS[backend], label=BACKEND_NAMES[backend])

    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Perfect Scaling')
    ax.set_xlabel('Concurrency Level')
    ax.set_ylabel('Scaling Efficiency (%)')
    ax.set_title('Scaling Efficiency vs Concurrency (50 tokens)\n(100% = Linear Scaling)')
    ax.legend()
    ax.set_xticks([1, 4, 8, 16])
    ax.set_ylim(0, 120)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'scaling_efficiency.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: scaling_efficiency.png")

def plot_summary_dashboard(df, gpu_metrics, output_dir):
    """Generate summary dashboard"""
    fig = plt.figure(figsize=(16, 12))

    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Peak Throughput (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    peak_throughput = df.groupby('backend')['throughput_rps'].max()
    colors = [COLORS[b] for b in peak_throughput.index]
    bars = ax1.bar([BACKEND_NAMES[b] for b in peak_throughput.index], peak_throughput.values, color=colors)
    ax1.set_ylabel('Requests/sec')
    ax1.set_title('Peak Throughput')
    for bar, val in zip(bars, peak_throughput.values):
        ax1.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, val),
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 2. Peak Tokens/sec (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    peak_tokens = df.groupby('backend')['tokens_per_sec'].max()
    bars = ax2.bar([BACKEND_NAMES[b] for b in peak_tokens.index], peak_tokens.values,
                   color=[COLORS[b] for b in peak_tokens.index])
    ax2.set_ylabel('Tokens/sec')
    ax2.set_title('Peak Token Rate')
    for bar, val in zip(bars, peak_tokens.values):
        ax2.annotate(f'{val:.0f}', xy=(bar.get_x() + bar.get_width()/2, val),
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 3. Best P95 Latency at c=1 (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    df_c1_50 = df[(df['concurrency'] == 1) & (df['max_tokens'] == 50)]
    p95_lat = df_c1_50.set_index('backend')['p95_latency_ms']
    bars = ax3.bar([BACKEND_NAMES[b] for b in p95_lat.index], p95_lat.values,
                   color=[COLORS[b] for b in p95_lat.index])
    ax3.set_ylabel('Latency (ms)')
    ax3.set_title('P95 Latency (c=1, 50 tokens)')
    for bar, val in zip(bars, p95_lat.values):
        ax3.annotate(f'{val:.0f}', xy=(bar.get_x() + bar.get_width()/2, val),
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 4. Throughput scaling (middle, spans 2 cols)
    ax4 = fig.add_subplot(gs[1, :2])
    df_50 = df[df['max_tokens'] == 50]
    for backend in ['vllm', 'triton', 'tgi']:
        backend_data = df_50[df_50['backend'] == backend].sort_values('concurrency')
        ax4.plot(backend_data['concurrency'], backend_data['throughput_rps'],
                marker='o', linewidth=2.5, markersize=10,
                color=COLORS[backend], label=BACKEND_NAMES[backend])
    ax4.set_xlabel('Concurrency')
    ax4.set_ylabel('Throughput (req/s)')
    ax4.set_title('Throughput Scaling (50 tokens)')
    ax4.legend(loc='upper left')
    ax4.set_xticks([1, 4, 8, 16])
    ax4.grid(True, alpha=0.3)

    # 5. GPU Utilization (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    if gpu_metrics:
        sm_data = []
        for backend in ['vllm', 'triton', 'tgi']:
            if backend in gpu_metrics and not gpu_metrics[backend].empty:
                active = gpu_metrics[backend][gpu_metrics[backend]['sm'] > 50]
                if not active.empty:
                    sm_data.append((BACKEND_NAMES[backend], active['sm'].mean(), COLORS[backend]))

        if sm_data:
            names, values, colors = zip(*sm_data)
            bars = ax5.bar(names, values, color=colors)
            ax5.set_ylabel('SM Utilization (%)')
            ax5.set_title('Avg GPU Utilization')
            ax5.set_ylim(0, 105)
            for bar, val in zip(bars, values):
                ax5.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, val),
                            ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 6. Latency comparison (bottom, spans 2 cols)
    ax6 = fig.add_subplot(gs[2, :2])
    for backend in ['vllm', 'triton', 'tgi']:
        backend_data = df_50[df_50['backend'] == backend].sort_values('concurrency')
        ax6.plot(backend_data['concurrency'], backend_data['p95_latency_ms'],
                marker='s', linewidth=2.5, markersize=10,
                color=COLORS[backend], label=BACKEND_NAMES[backend])
    ax6.set_xlabel('Concurrency')
    ax6.set_ylabel('P95 Latency (ms)')
    ax6.set_title('P95 Latency Scaling (50 tokens)')
    ax6.legend(loc='upper left')
    ax6.set_xticks([1, 4, 8, 16])
    ax6.grid(True, alpha=0.3)

    # 7. Winner summary (bottom right)
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')

    # Determine winners
    winners = {
        'Peak Throughput': peak_throughput.idxmax(),
        'Peak Tokens/s': peak_tokens.idxmax(),
        'Lowest Latency': p95_lat.idxmin(),
    }

    if gpu_metrics:
        sm_avg = {b: gpu_metrics[b][gpu_metrics[b]['sm'] > 50]['sm'].mean()
                  for b in gpu_metrics if not gpu_metrics[b].empty and len(gpu_metrics[b][gpu_metrics[b]['sm'] > 50]) > 0}
        if sm_avg:
            winners['Best GPU Util'] = max(sm_avg, key=sm_avg.get)

    summary_text = "WINNERS\n" + "="*30 + "\n\n"
    for category, winner in winners.items():
        summary_text += f"{category}:\n  {BACKEND_NAMES[winner]}\n\n"

    ax7.text(0.1, 0.9, summary_text, transform=ax7.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('LLM Inference Benchmark Summary\nvLLM vs NVIDIA Triton vs HuggingFace TGI',
                fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(output_dir / 'benchmark_summary_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: benchmark_summary_dashboard.png")

def plot_heatmap_comparison(df, output_dir):
    """Generate heatmap comparison"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, metric in enumerate([('throughput_rps', 'Throughput (req/s)'),
                                   ('tokens_per_sec', 'Tokens/sec'),
                                   ('p95_latency_ms', 'P95 Latency (ms)')]):
        ax = axes[idx]
        metric_name, metric_label = metric

        # Create pivot table
        pivot = df.pivot_table(values=metric_name,
                               index='backend',
                               columns=['concurrency', 'max_tokens'],
                               aggfunc='mean')

        # Reorder backends
        pivot = pivot.reindex(['vllm', 'triton', 'tgi'])
        pivot.index = [BACKEND_NAMES[b] for b in pivot.index]

        # Create heatmap
        im = ax.imshow(pivot.values, cmap='RdYlGn' if 'Latency' not in metric_label else 'RdYlGn_r',
                       aspect='auto')

        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Set labels
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f'c={c[0]}\nt={c[1]}' for c in pivot.columns], fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title(metric_label)

        # Add value annotations
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                text = f'{val:.1f}' if val < 100 else f'{val:.0f}'
                ax.text(j, i, text, ha='center', va='center', fontsize=8)

    plt.suptitle('Performance Heatmap Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: performance_heatmap.png")

def main():
    # Paths
    base_dir = Path("/Users/deepsoni/Oracle Content - Accounts/Oracle Content/Projects/CoE_initial/CoE_base/000-AICoE-Knowledgebase/001-customers/2025-lab/LLM-training/vLLMvsTritonvsHFTGI")
    benchmark_dir = base_dir / "results" / "benchmarks" / "20260108"
    profile_dir = base_dir / "results" / "profiles" / "20260108_final"
    output_dir = base_dir / "results" / "plots"

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("  LLM Inference Benchmark Visualization")
    print("="*60)
    print(f"\nBenchmark data: {benchmark_dir}")
    print(f"GPU profiles: {profile_dir}")
    print(f"Output: {output_dir}")
    print()

    # Load data
    print("Loading benchmark data...")
    df = load_benchmark_data(benchmark_dir)
    print(f"  Loaded {len(df)} benchmark results")

    print("Loading GPU metrics...")
    gpu_metrics = load_gpu_metrics(profile_dir)
    print(f"  Loaded metrics for: {list(gpu_metrics.keys())}")

    print("\nGenerating visualizations...")

    # Generate plots
    plot_throughput_comparison(df, output_dir)
    plot_tokens_per_second(df, output_dir)
    plot_latency_comparison(df, output_dir)
    plot_scaling_efficiency(df, output_dir)
    plot_heatmap_comparison(df, output_dir)

    if gpu_metrics:
        plot_gpu_metrics(gpu_metrics, output_dir)

    plot_summary_dashboard(df, gpu_metrics, output_dir)

    print("\n" + "="*60)
    print("  Visualization Complete!")
    print("="*60)
    print(f"\nAll plots saved to: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  - {f.name}")

if __name__ == "__main__":
    main()
