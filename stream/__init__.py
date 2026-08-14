"""Layer/expert streaming for TurboQuant-MLX.

Runs MoE models far larger than RAM by keeping only the small resident
tensors (norms, routers, attention, shared experts, embeddings) in memory
and streaming the router-selected experts from an mmap'd safetensors file
on demand, with an LRU cache.

Status: proof-of-concept (qwen3_5_moe).
"""
