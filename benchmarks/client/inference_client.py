#!/usr/bin/env python3
"""
Universal LLM Inference Client
Supports vLLM, NVIDIA Triton, and HuggingFace TGI backends
with identical prompts for fair comparison.

Usage:
  python3 inference_client.py --backend vllm --iterations 100
  python3 inference_client.py --backend triton --concurrency 8
  python3 inference_client.py --backend tgi --output-tokens 512
"""

import argparse
import json
import time
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from typing import Dict, List, Optional, Tuple

# Default configuration
DEFAULT_PROMPTS = [
    "Explain the concept of neural networks in simple terms.",
    "Write a Python function to calculate the Fibonacci sequence.",
    "What are the key differences between SQL and NoSQL databases?",
    "Describe the water cycle in detail.",
    "Explain how transformers work in deep learning.",
]

BACKEND_CONFIGS = {
    "vllm": {
        "endpoint": "http://localhost:8000/v1/completions",
        "health": "http://localhost:8000/health",
        "format": "openai",
    },
    "triton": {
        "endpoint": "http://localhost:8000/v2/models/mistral/generate",
        "health": "http://localhost:8000/v2/health/ready",
        "format": "triton",
    },
    "tgi": {
        "endpoint": "http://localhost:8000/generate",
        "health": "http://localhost:8000/health",
        "format": "tgi",
    },
}


def format_request(backend: str, prompt: str, max_tokens: int = 128) -> Tuple[str, dict]:
    """Format request based on backend API."""
    config = BACKEND_CONFIGS[backend]

    if config["format"] == "openai":
        # vLLM OpenAI-compatible format
        data = {
            "model": "mistralai/Mistral-7B-Instruct-v0.2",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
        }
    elif config["format"] == "triton":
        # Triton vLLM backend format
        data = {
            "text_input": prompt,
            "parameters": {
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 0.9,
            }
        }
    elif config["format"] == "tgi":
        # HuggingFace TGI format
        data = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
            }
        }
    else:
        raise ValueError(f"Unknown format: {config['format']}")

    return config["endpoint"], data


def parse_response(backend: str, response_data: dict) -> Tuple[str, int]:
    """Parse response and extract generated text and token count."""
    config = BACKEND_CONFIGS[backend]

    if config["format"] == "openai":
        text = response_data["choices"][0]["text"]
        tokens = response_data.get("usage", {}).get("completion_tokens", len(text.split()))
    elif config["format"] == "triton":
        text = response_data.get("text_output", "")
        tokens = len(text.split())  # Approximate
    elif config["format"] == "tgi":
        text = response_data.get("generated_text", "")
        tokens = response_data.get("details", {}).get("generated_tokens", len(text.split()))
    else:
        text = str(response_data)
        tokens = len(text.split())

    return text, tokens


def check_health(backend: str, timeout: int = 5) -> bool:
    """Check if backend is healthy."""
    config = BACKEND_CONFIGS[backend]
    try:
        req = Request(config["health"])
        with urlopen(req, timeout=timeout) as response:
            return response.status == 200
    except Exception:
        return False


def single_inference(backend: str, prompt: str, max_tokens: int, timeout: int = 60) -> Dict:
    """Execute a single inference request."""
    endpoint, data = format_request(backend, prompt, max_tokens)

    result = {
        "success": False,
        "latency_ms": 0,
        "tokens_generated": 0,
        "ttft_ms": 0,  # Time to first token (approximated)
        "error": None,
    }

    try:
        req = Request(endpoint)
        req.add_header("Content-Type", "application/json")
        payload = json.dumps(data).encode("utf-8")

        start_time = time.perf_counter()
        with urlopen(req, data=payload, timeout=timeout) as response:
            response_data = json.loads(response.read().decode("utf-8"))
        end_time = time.perf_counter()

        text, tokens = parse_response(backend, response_data)

        result["success"] = True
        result["latency_ms"] = (end_time - start_time) * 1000
        result["tokens_generated"] = tokens
        result["ttft_ms"] = result["latency_ms"] / 2  # Rough approximation

    except HTTPError as e:
        result["error"] = f"HTTP {e.code}: {e.reason}"
    except URLError as e:
        result["error"] = f"URL Error: {e.reason}"
    except Exception as e:
        result["error"] = str(e)[:200]

    return result


def run_benchmark(
    backend: str,
    prompts: List[str],
    iterations: int,
    warmup: int,
    max_tokens: int,
    concurrency: int,
    quiet: bool = False
) -> Dict:
    """Run benchmark with given configuration."""

    results = {
        "backend": backend,
        "config": {
            "iterations": iterations,
            "warmup": warmup,
            "max_tokens": max_tokens,
            "concurrency": concurrency,
        },
        "latencies_ms": [],
        "tokens_generated": [],
        "errors": 0,
        "total_time_sec": 0,
    }

    if not quiet:
        print(f"\n{'='*60}")
        print(f"  Benchmark: {backend.upper()}")
        print(f"{'='*60}")
        print(f"  Iterations: {iterations}")
        print(f"  Warmup: {warmup}")
        print(f"  Max Tokens: {max_tokens}")
        print(f"  Concurrency: {concurrency}")
        print()

    # Check health
    if not check_health(backend):
        print(f"ERROR: {backend} backend not ready")
        results["error"] = "Backend not ready"
        return results

    if not quiet:
        print(f"Backend ready.\n")

    # Warmup
    if warmup > 0 and not quiet:
        print(f"Warming up ({warmup} requests)...")
        for i in range(warmup):
            prompt = prompts[i % len(prompts)]
            single_inference(backend, prompt, max_tokens)
        print("Warmup complete.\n")

    # Benchmark
    if not quiet:
        print(f"Running benchmark ({iterations} requests)...\n")

    start_time = time.perf_counter()

    if concurrency == 1:
        # Sequential execution
        for i in range(iterations):
            prompt = prompts[i % len(prompts)]
            result = single_inference(backend, prompt, max_tokens)

            if result["success"]:
                results["latencies_ms"].append(result["latency_ms"])
                results["tokens_generated"].append(result["tokens_generated"])
            else:
                results["errors"] += 1
                if results["errors"] == 1 and not quiet:
                    print(f"  First error: {result['error']}")

            if not quiet and (i + 1) % 10 == 0:
                avg_lat = sum(results["latencies_ms"]) / len(results["latencies_ms"]) if results["latencies_ms"] else 0
                print(f"  Progress: {i+1}/{iterations} - Avg latency: {avg_lat:.1f} ms")
    else:
        # Concurrent execution
        def worker(idx):
            prompt = prompts[idx % len(prompts)]
            return single_inference(backend, prompt, max_tokens)

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(worker, i) for i in range(iterations)]
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()

                if result["success"]:
                    results["latencies_ms"].append(result["latency_ms"])
                    results["tokens_generated"].append(result["tokens_generated"])
                else:
                    results["errors"] += 1
                    if results["errors"] == 1 and not quiet:
                        print(f"  First error: {result['error']}")

                if not quiet and i % max(1, iterations // 10) == 0:
                    avg_lat = sum(results["latencies_ms"]) / len(results["latencies_ms"]) if results["latencies_ms"] else 0
                    print(f"  Progress: {i}/{iterations} - Avg latency: {avg_lat:.1f} ms - Errors: {results['errors']}")

    end_time = time.perf_counter()
    results["total_time_sec"] = end_time - start_time

    return results


def calculate_statistics(results: Dict) -> Dict:
    """Calculate benchmark statistics."""
    latencies = results["latencies_ms"]
    tokens = results["tokens_generated"]

    if not latencies:
        return {"error": "No successful requests"}

    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)

    total_tokens = sum(tokens)
    total_time = results["total_time_sec"]

    stats = {
        "successful_requests": n,
        "failed_requests": results["errors"],
        "error_rate_percent": (results["errors"] / results["config"]["iterations"]) * 100,
        "total_time_sec": total_time,
        "latency_ms": {
            "min": min(latencies),
            "max": max(latencies),
            "mean": sum(latencies) / n,
            "median": latencies_sorted[n // 2],
            "p50": latencies_sorted[int(n * 0.50)],
            "p90": latencies_sorted[int(n * 0.90)],
            "p95": latencies_sorted[int(n * 0.95)],
            "p99": latencies_sorted[min(int(n * 0.99), n - 1)],
        },
        "throughput": {
            "requests_per_sec": n / total_time,
            "tokens_per_sec": total_tokens / total_time,
        },
        "tokens": {
            "total_generated": total_tokens,
            "avg_per_request": total_tokens / n,
        },
    }

    return stats


def print_results(results: Dict, stats: Dict):
    """Print benchmark results."""
    print(f"\n{'='*60}")
    print(f"  Results: {results['backend'].upper()}")
    print(f"{'='*60}\n")

    print(f"Requests:")
    print(f"  Successful: {stats['successful_requests']}")
    print(f"  Failed:     {stats['failed_requests']}")
    print(f"  Error Rate: {stats['error_rate_percent']:.1f}%")
    print()

    print(f"Latency (ms):")
    print(f"  Min:    {stats['latency_ms']['min']:8.2f}")
    print(f"  Max:    {stats['latency_ms']['max']:8.2f}")
    print(f"  Mean:   {stats['latency_ms']['mean']:8.2f}")
    print(f"  Median: {stats['latency_ms']['median']:8.2f}")
    print(f"  P50:    {stats['latency_ms']['p50']:8.2f}")
    print(f"  P90:    {stats['latency_ms']['p90']:8.2f}")
    print(f"  P95:    {stats['latency_ms']['p95']:8.2f}")
    print(f"  P99:    {stats['latency_ms']['p99']:8.2f}")
    print()

    print(f"Throughput:")
    print(f"  Requests/sec: {stats['throughput']['requests_per_sec']:8.2f}")
    print(f"  Tokens/sec:   {stats['throughput']['tokens_per_sec']:8.2f}")
    print()

    print(f"Tokens:")
    print(f"  Total:       {stats['tokens']['total_generated']}")
    print(f"  Avg/request: {stats['tokens']['avg_per_request']:.1f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="LLM Inference Benchmark Client")
    parser.add_argument("--backend", choices=["vllm", "triton", "tgi"], required=True,
                       help="Inference backend to benchmark")
    parser.add_argument("--iterations", type=int, default=100,
                       help="Number of inference requests (default: 100)")
    parser.add_argument("--warmup", type=int, default=10,
                       help="Number of warmup requests (default: 10)")
    parser.add_argument("--max-tokens", type=int, default=128,
                       help="Maximum tokens to generate (default: 128)")
    parser.add_argument("--concurrency", type=int, default=1,
                       help="Number of concurrent workers (default: 1)")
    parser.add_argument("--prompts-file", type=str, default=None,
                       help="JSON file with prompts (default: use built-in)")
    parser.add_argument("--output-file", type=str, default=None,
                       help="Output JSON file for results")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress progress output")

    args = parser.parse_args()

    # Load prompts
    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = json.load(f)
    else:
        prompts = DEFAULT_PROMPTS

    # Run benchmark
    results = run_benchmark(
        backend=args.backend,
        prompts=prompts,
        iterations=args.iterations,
        warmup=args.warmup,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        quiet=args.quiet,
    )

    # Calculate statistics
    stats = calculate_statistics(results)

    # Print results
    if not args.quiet:
        print_results(results, stats)

    # Save results
    if args.output_file:
        output_data = {
            "backend": args.backend,
            "config": results["config"],
            "statistics": stats,
            "raw_latencies": results["latencies_ms"],
            "raw_tokens": results["tokens_generated"],
        }

        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        if not args.quiet:
            print(f"Results saved to: {output_path}")

    # Return success/failure
    return 0 if stats.get("successful_requests", 0) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
