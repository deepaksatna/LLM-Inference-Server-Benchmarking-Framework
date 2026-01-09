# Contributing to LLM Inference Benchmarking Framework

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Submitting Changes](#submitting-changes)

---

## Code of Conduct

This project follows a standard code of conduct. Please be respectful and constructive in all interactions.

---

## Getting Started

### Prerequisites

- Python 3.10+
- Docker
- kubectl configured with a Kubernetes cluster
- Access to NVIDIA GPU nodes

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/llm-inference-benchmark.git
   cd llm-inference-benchmark
   ```
3. Add upstream remote:
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/llm-inference-benchmark.git
   ```

---

## How to Contribute

### Reporting Issues

- Use the GitHub issue tracker
- Include:
  - Clear description of the issue
  - Steps to reproduce
  - Expected vs actual behavior
  - Environment details (GPU, K8s version, etc.)

### Suggesting Features

- Open a GitHub issue with the "enhancement" label
- Describe the feature and its use case
- Explain how it would benefit users

### Contributing Code

1. Check existing issues and PRs
2. Create an issue for discussion (for large changes)
3. Fork and create a feature branch
4. Make your changes
5. Submit a pull request

---

## Development Setup

### Local Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=benchmarks

# Run specific tests
pytest tests/test_benchmark.py
```

---

## Coding Standards

### Python Style

- Follow PEP 8
- Use type hints
- Maximum line length: 100 characters
- Use meaningful variable names

```python
# Good
def calculate_throughput(
    total_requests: int,
    duration_seconds: float
) -> float:
    """Calculate requests per second."""
    return total_requests / duration_seconds

# Avoid
def calc(r, d):
    return r / d
```

### Formatting

```bash
# Format code
black .

# Sort imports
isort .

# Type checking
mypy benchmarks/
```

### YAML Style

- Use 2-space indentation
- Include comments for non-obvious configurations
- Use meaningful resource names

```yaml
# Good
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server  # Clear, descriptive name
  labels:
    app: vllm
    component: inference-server
```

### Shell Scripts

- Use `set -e` for error handling
- Include usage comments
- Quote variables

```bash
#!/bin/bash
set -e

# Usage: ./deploy.sh <backend>
# Deploys the specified inference backend

BACKEND="${1:-vllm}"  # Default to vllm

if [ -z "$BACKEND" ]; then
    echo "Usage: $0 <backend>"
    exit 1
fi
```

---

## Project Structure

When adding new features, follow the existing structure:

```
llm-inference-benchmark/
├── docker/<server>/         # Add new server Docker files here
├── k8s/<server>/           # Add new server K8s manifests here
├── benchmarks/
│   ├── client/             # Benchmark client code
│   └── configs/            # Configuration files
├── scripts/                # Automation scripts
└── tests/                  # Test files
```

### Adding a New Inference Server

1. **Create Docker image**:
   ```
   docker/<new-server>/
   ├── Dockerfile
   └── any_helper_scripts.sh
   ```

2. **Create K8s deployment**:
   ```
   k8s/<new-server>/
   └── deployment.yaml
   ```

3. **Add benchmark support**:
   - Update `benchmarks/client/inference_client.py`
   - Add backend type handling

4. **Update configurations**:
   - Add to `benchmarks/configs/benchmark_matrix.yaml`

5. **Add tests**:
   ```
   tests/test_<new-server>.py
   ```

6. **Update documentation**:
   - Add to README.md
   - Include in comparison tables

### Adding a New GPU Configuration

1. **Create results directory**:
   ```
   results/benchmarks/<GPU>-<Model>/
   ```

2. **Run benchmarks** with appropriate configs

3. **Update GPU compatibility table** in README.md

---

## Submitting Changes

### Pull Request Process

1. **Create feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes** and commit:
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   ```

3. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Open Pull Request** on GitHub

### Commit Message Format

Use conventional commits:

```
<type>: <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

Examples:
```
feat: add support for Llama 3 models

fix: correct throughput calculation for streaming responses

docs: update GPU compatibility table with H200 specs
```

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated (if needed)
- [ ] Commit messages follow format
- [ ] PR description explains changes

---

## Questions?

- Open a GitHub issue for questions
- Tag maintainers for urgent matters

Thank you for contributing!
