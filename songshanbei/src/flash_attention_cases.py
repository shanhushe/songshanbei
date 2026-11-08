"""Check the staged NineToothed InfiniCore kernel on model-relevant shapes."""

import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from infinicore.nn.functional.flash_attention import flash_attention
from infinicore.tensor import from_torch


def check(label, query_heads, kv_heads, query_length, key_length, causal, *, width=128, mha=None):
    dtype = torch.bfloat16
    device = "cuda"
    query = torch.randn((1, query_heads, query_length, width), dtype=dtype, device=device)
    key = torch.randn((1, kv_heads, key_length, width), dtype=dtype, device=device)
    value = torch.randn_like(key)
    total_kv = torch.full((16,), key_length, dtype=torch.int32, device=device)
    scale = 1 / math.sqrt(width)
    if mha is None:
        result = flash_attention(
            from_torch(query), from_torch(key), from_torch(value), from_torch(total_kv),
            is_causal=causal, scale=scale,
        )
    else:
        bshd = [tensor.transpose(1, 2).contiguous() for tensor in (query, key, value)]
        result = mha(*(from_torch(tensor) for tensor in bshd), is_causal=causal, scale=scale)
    with TemporaryDirectory(prefix="h3-attention-result-") as directory:
        output = Path(directory) / "attention.bin"
        result.debug(str(output))
        elements = query.numel()
        if output.stat().st_size == elements * 2:
            values = torch.from_file(str(output), size=elements, dtype=torch.int16)
            values = values.view(torch.bfloat16).float()
        else:
            assert output.stat().st_size == elements * 4, label
            values = torch.from_file(str(output), size=elements, dtype=torch.float32)
        if mha is None:
            result_tensor = values.reshape(query.shape).to(device)
        else:
            result_tensor = values.reshape(bshd[0].shape).transpose(1, 2).to(device)
    groups = query_heads // kv_heads
    reference_key = key.repeat_interleave(groups, dim=1).float()
    reference_value = value.repeat_interleave(groups, dim=1).float()
    logits = (query.float() @ reference_key.transpose(-1, -2)) * scale
    if causal:
        query_index = torch.arange(query_length, device=device)[:, None]
        key_index = torch.arange(key_length, device=device)[None, :]
        logits.masked_fill_(key_index > query_index + key_length - query_length, -torch.inf)
    expected = logits.softmax(dim=-1) @ reference_value
    difference = (result_tensor - expected).abs()
    relative_l2 = (result_tensor - expected).norm() / expected.norm()
    print(
        f"{label} max_abs={difference.max().item():.6f} relative_l2={relative_l2.item():.6f}",
        flush=True,
    )
    assert torch.isfinite(result_tensor).all().item(), f"{label} non-finite output"
    assert difference.max().item() < 0.05 and relative_l2.item() < 0.04, label


if __name__ == "__main__":
    torch.manual_seed(42)
    if "--mha" in sys.argv:
        from infinicore.ops.mha import mha
        from launch_nine import register

        bridge = register()
        check("hygon_mha_128", 8, 2, 129, 129, True, mha=mha)
        check("hygon_mha_72", 8, 8, 129, 129, False, width=72, mha=mha)
    else:
        check("bf16_aligned", 8, 8, 256, 256, False)
        check("bf16_tail", 8, 8, 129, 257, False)
        check("bf16_gqa", 8, 2, 129, 257, False)
        if "--causal" in sys.argv:
            check("bf16_causal", 8, 2, 129, 129, True)
    print("MODEL_SHAPES_PASS", flush=True)
