"""Exercise the Nine-generated launcher directly against a BF16 reference."""

import ctypes as ct
import math
import sys

import torch


class NineTensor(ct.Structure):
    _fields_ = [
        ("data", ct.c_void_p),
        ("shape", ct.POINTER(ct.c_uint64)),
        ("strides", ct.POINTER(ct.c_int64)),
    ]

    @classmethod
    def from_tensor(cls, tensor):
        return cls(
            tensor.data_ptr(),
            (ct.c_uint64 * tensor.ndim)(*tensor.shape),
            (ct.c_int64 * tensor.ndim)(*tensor.stride()),
        )


def check(launch, name, query_heads, kv_heads, query_length, kv_length, causal, variant, padding):
    device = "cuda"
    dtype = torch.bfloat16
    query = torch.randn(1, query_heads, query_length, 128, device=device, dtype=dtype)
    key = torch.randn(1, kv_heads, kv_length, 128, device=device, dtype=dtype)
    value = torch.randn_like(key)
    pad = torch.nn.functional.pad
    launch_query = pad(query, (0, 0, 0, (-query_length) % 64)) if padding else query
    launch_key = pad(key, (0, 0, 0, (-kv_length) % 64)) if padding == "both" else key
    launch_value = pad(value, (0, 0, 0, (-kv_length) % 64)) if padding == "both" else value
    out = torch.full_like(launch_query, torch.nan)
    lengths = torch.full((16,), kv_length, dtype=torch.int32, device=device)
    scale = ct.c_float(1 / math.sqrt(128))
    scale_arg = NineTensor(ct.addressof(scale), None, None)
    mask = NineTensor(None, (ct.c_uint64 * 4)(16, 16, 16, 16), (ct.c_int64 * 4)(1, 1, 1, 1))
    unused = NineTensor(None, None, None)
    print(f"AOT_CASE_START {name}", flush=True)
    rc = launch(
        torch.cuda.current_stream().cuda_stream,
        *(NineTensor.from_tensor(tensor) for tensor in (launch_query, launch_key, launch_value, lengths)),
        mask, unused, scale_arg, NineTensor.from_tensor(out), unused, unused,
        0, 128, causal, 0, variant, 19, 64, 64,
    )
    print(f"AOT_LAUNCH name={name} result={rc}", flush=True)
    assert rc == 0, name
    torch.cuda.synchronize()
    repeat = query_heads // kv_heads
    expanded_key = key.repeat_interleave(repeat, dim=1).float()
    expanded_value = value.repeat_interleave(repeat, dim=1).float()
    logits = (query.float() @ expanded_key.transpose(-1, -2)) * scale.value
    if causal:
        q_positions = torch.arange(query_length, device=device)[:, None]
        k_positions = torch.arange(kv_length, device=device)[None, :]
        logits.masked_fill_(k_positions > q_positions, -torch.inf)
    expected = logits.softmax(-1) @ expanded_value
    result = out[:, :, :query_length].float()
    error = result - expected
    max_abs = error.abs().max().item()
    relative_l2 = (error.norm() / expected.norm()).item()
    worst_row = error.abs().amax(dim=(0, 1, 3)).argmax().item()
    print(f"AOT_NUMERIC name={name} max_abs={max_abs:.6f} relative_l2={relative_l2:.6f} worst_row={worst_row}", flush=True)
    assert torch.isfinite(result).all().item() and max_abs < 0.05 and relative_l2 < 0.04, name


def main(path, causal, variant):
    library = ct.CDLL(path)
    launch = library.launch_flash_attention
    launch.argtypes = [ct.c_void_p] + [NineTensor] * 10 + [ct.c_int] * 8
    launch.restype = ct.c_int
    torch.manual_seed(42)
    check(launch, "aligned", 8, 8, 256, 256, causal, variant, None)
    kv_length = 129 if causal else 257
    check(launch, "tail", 8, 8, 129, kv_length, causal, variant, None)
    check(launch, "gqa", 8, 2, 129, kv_length, causal, variant, None)
    print("DIRECT_AOT_NUMERIC_PASS", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], "--causal" in sys.argv[2:], 1)
