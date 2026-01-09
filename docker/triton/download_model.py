#!/usr/bin/env python3
"""Download Mistral model for offline use."""
from huggingface_hub import snapshot_download
import os

model_id = 'mistralai/Mistral-7B-Instruct-v0.2'
cache_dir = '/root/.cache/huggingface/hub'

print(f'Downloading {model_id}...')
snapshot_download(
    repo_id=model_id,
    cache_dir=cache_dir,
    token=os.environ.get('HF_TOKEN'),
    local_dir_use_symlinks=False
)
print('Model download complete!')
