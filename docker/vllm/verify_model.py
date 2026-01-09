#!/usr/bin/env python3
"""Verify model is accessible offline."""
from transformers import AutoTokenizer, AutoConfig

print('Verifying offline model access...')
tokenizer = AutoTokenizer.from_pretrained(
    'mistralai/Mistral-7B-Instruct-v0.2',
    local_files_only=True
)
config = AutoConfig.from_pretrained(
    'mistralai/Mistral-7B-Instruct-v0.2',
    local_files_only=True
)
print(f'Model: {config.model_type}')
print(f'Vocab size: {tokenizer.vocab_size}')
print('Offline verification complete!')
