"""兼容补丁 v6（放到 ~/pylibs/sitecustomize.py，Python 启动时自动加载）。

v6 = v5 的全部补丁 + **长序列注意力分块**（关键新增）。

为什么要分块：
  DTK 镜像里的 torch 没有 flash_attn_2_cuda*.so，SDPA 的 flash 后端加载失败；
  mem-efficient(CK) 后端在 gfx936 上不一定可用，退化成 math 内核后，
  视频 transformer 的注意力要一次性开 O(N^2) 的分数矩阵 —— 124 帧时是 158 GiB，必爆。

分块后：把 query 按行切成 H3_ATTN_CHUNK 行一块，逐块算 SDPA，
峰值显存从 O(N^2) 降到 O(chunk*N)，结果完全等价（各块独立，非因果注意力）。

开关（可选，默认已在前面设置好）：
  H3_ATTN_CHUNK         每块行数，默认 2048
  H3_ATTN_TRIGGER       序列长度超过多少才分块，默认 4096
  H3_FORCE_EFFICIENT=1  强制只用 mem-efficient 后端（若可用），不走 math
"""

import os
import sys


def _log(msg):
    print(f"[sitecustomize] {msg}", file=sys.stderr)


# --------------------------------------------------------------------------
# 0) SDPA 后端：关 flash（没有 .so），保留 mem-efficient 与 math
# --------------------------------------------------------------------------
try:
    import torch

    torch.backends.cuda.enable_flash_sdp(False)
    try:
        torch.backends.cuda.enable_cudnn_sdp(False)
    except Exception:
        pass
    try:
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    except Exception:
        pass
    if os.environ.get("H3_FORCE_EFFICIENT", "") == "1":
        torch.backends.cuda.enable_math_sdp(False)
        _log("SDPA: 强制 mem-efficient，math 关闭")
    else:
        torch.backends.cuda.enable_math_sdp(True)
    _log(
        "SDPA: flash=%s mem_efficient=%s math=%s"
        % (
            torch.backends.cuda.flash_sdp_enabled(),
            torch.backends.cuda.mem_efficient_sdp_enabled(),
            torch.backends.cuda.math_sdp_enabled(),
        )
    )
except Exception as exc:
    _log(f"step0 failed: {exc}")


# --------------------------------------------------------------------------
# 0.5) 长序列注意力分块：包住 F.scaled_dot_product_attention
# --------------------------------------------------------------------------
try:
    import torch
    import torch.nn.functional as F

    _raw_sdpa = F.scaled_dot_product_attention

    def _orig_sdpa(*args, **kwargs):
        """兼容老版本 torch：没有 enable_gqa 关键字时自动去掉重试。"""
        try:
            return _raw_sdpa(*args, **kwargs)
        except TypeError:
            kwargs.pop("enable_gqa", None)
            return _raw_sdpa(*args, **kwargs)

    _CHUNK = int(os.environ.get("H3_ATTN_CHUNK", "2048"))
    _TRIGGER = int(os.environ.get("H3_ATTN_TRIGGER", "4096"))
    _state = {"chunked": 0}

    def _chunked_sdpa(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
    ):
        try:
            seq_q = query.shape[-2]
            seq_k = key.shape[-2]
        except Exception:
            return _orig_sdpa(query, key, value, attn_mask, dropout_p,
                              is_causal, scale=scale, enable_gqa=enable_gqa)

        # mask 能否按 query 行切
        mask_rows_ok = False
        if attn_mask is not None:
            try:
                mask_rows_ok = (attn_mask.dim() >= 2 and attn_mask.shape[-2] in (1, seq_q)) and not is_causal
            except Exception:
                mask_rows_ok = False

        # 只有"超长 + 推理(无 dropout) + mask 可切/无 mask"才分块
        if (
            dropout_p != 0.0
            or seq_q <= _TRIGGER
            or _CHUNK <= 0
            or (attn_mask is not None and not mask_rows_ok)
        ):
            return _orig_sdpa(query, key, value, attn_mask, dropout_p,
                              is_causal, scale=scale, enable_gqa=enable_gqa)

        if _state["chunked"] == 0:
            _log(f"sdpa 分块启用: seq={seq_q}x{seq_k}, chunk={_CHUNK}, "
                 f"mask={'yes' if attn_mask is not None else 'no'}")
        _state["chunked"] += 1

        outs = []
        for start in range(0, seq_q, _CHUNK):
            end = min(start + _CHUNK, seq_q)
            q_blk = query[..., start:end, :]
            if is_causal:
                # 该块第 i 行（绝对下标 start+i）可见 0..start+i 列
                rows = torch.arange(start, end, device=query.device).unsqueeze(1)
                cols = torch.arange(seq_k, device=query.device).unsqueeze(0)
                outs.append(
                    _orig_sdpa(q_blk, key, value, cols <= rows, dropout_p,
                               False, scale=scale, enable_gqa=enable_gqa)
                )
            elif attn_mask is not None:
                m_blk = attn_mask if attn_mask.shape[-2] == 1 else attn_mask[..., start:end, :]
                outs.append(
                    _orig_sdpa(q_blk, key, value, m_blk, dropout_p,
                               False, scale=scale, enable_gqa=enable_gqa)
                )
            else:
                outs.append(
                    _orig_sdpa(q_blk, key, value, None, dropout_p,
                               False, scale=scale, enable_gqa=enable_gqa)
                )
        import torch as _t
        return _t.cat(outs, dim=-2)

    F.scaled_dot_product_attention = _chunked_sdpa
    try:
        torch.nn.functional.scaled_dot_product_attention = _chunked_sdpa
    except Exception:
        pass
    _log(f"scaled_dot_product_attention 已包装 (trigger={_TRIGGER}, chunk={_CHUNK})")
except Exception as exc:
    _log(f"step0.5 failed: {exc}")


# --------------------------------------------------------------------------
# 1) Qwen2VL Fast 图像处理器 -> PIL 版（不依赖 torchvision）
# --------------------------------------------------------------------------
try:
    from transformers.models.qwen2_vl import image_processing_qwen2_vl as _fast_mod
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import (
        Qwen2VLImageProcessorPil,
    )

    _fast_mod.Qwen2VLImageProcessor = Qwen2VLImageProcessorPil
    _log("Qwen2VLImageProcessor -> PIL")
except Exception as exc:
    _log(f"step1 failed: {exc}")


# --------------------------------------------------------------------------
# 2) AutoImageProcessor 强制走 PIL 后端
# --------------------------------------------------------------------------
try:
    from transformers import AutoImageProcessor

    _orig_auto = AutoImageProcessor.from_pretrained.__func__

    def _auto_from_pretrained(cls, *args, **kwargs):
        kwargs.setdefault("use_fast", False)
        kwargs.setdefault("backend", "pil")
        return _orig_auto(cls, *args, **kwargs)

    AutoImageProcessor.from_pretrained = classmethod(_auto_from_pretrained)
    _log("AutoImageProcessor -> pil backend")
except Exception as exc:
    _log(f"step2 failed: {exc}")


# --------------------------------------------------------------------------
# 3) 允许 None 子处理器（video_processor 缺失）
# --------------------------------------------------------------------------
try:
    from transformers.processing_utils import ProcessorMixin

    _orig_check = ProcessorMixin.check_argument_for_proper_class

    def _check(self, argument_name, argument):
        if argument is None:
            return
        return _orig_check(self, argument_name, argument)

    ProcessorMixin.check_argument_for_proper_class = _check
    _log("None sub-processor allowed")
except Exception as exc:
    _log(f"step3 failed: {exc}")


# --------------------------------------------------------------------------
# 4) Qwen3VLProcessor 退化构造
# --------------------------------------------------------------------------
try:
    from transformers import AutoTokenizer, Qwen3VLProcessor

    _orig_qwen = Qwen3VLProcessor.from_pretrained.__func__

    def _qwen_from_pretrained(cls, path, *args, **kwargs):
        try:
            return _orig_qwen(cls, path, *args, **kwargs)
        except ImportError:
            sub = kwargs.get("subfolder")
            real = os.path.join(path, sub) if sub else path
            _log(f"fallback processor: {real}")
            tokenizer = AutoTokenizer.from_pretrained(real)
            image_processor = Qwen2VLImageProcessorPil.from_pretrained(real)
            return cls(
                image_processor=image_processor,
                tokenizer=tokenizer,
                video_processor=None,
            )

    Qwen3VLProcessor.from_pretrained = classmethod(_qwen_from_pretrained)
    _log("Qwen3VLProcessor fallback installed")
except Exception as exc:
    _log(f"step4 failed: {exc}")