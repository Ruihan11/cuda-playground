import torch
import torch.cuda.nvtx as nvtx
from flash_attn.cute import flash_attn_func

B, S, H, D = 2, 4096, 16, 128
Q = torch.randn(B, S, H, D, dtype=torch.float16, device="cuda")
K = torch.randn(B, S, H, D, dtype=torch.float16, device="cuda")
V = torch.randn(B, S, H, D, dtype=torch.float16, device="cuda")

for _ in range(10):
    flash_attn_func(Q, K, V)
torch.cuda.synchronize()

nvtx.range_push("fa4_fwd")
for _ in range(20):
    flash_attn_func(Q, K, V)
torch.cuda.synchronize()
nvtx.range_pop()
