"""Prepare the validated noncausal Nine kernel and its causal counterpart."""

import re
import sys
from pathlib import Path


def replace_once(text, pattern, replacement, label):
    changed, count = re.subn(pattern, replacement, text)
    if count != 1:
        raise ValueError(f"{label}: expected one match, found {count}")
    return changed


def causal_generator(source, target):
    text = source.read_text()
    text = replace_once(text, r'("block_size_m"\s*:\s*\()256(,\))', r"\g<1>64\2", "tile")
    text = replace_once(text, r'("is_causal"\s*:\s*\()0(,\))', r"\g<1>1\2", "causal")
    text = replace_once(
        text, r'op_name="flash_attention"', 'op_name="flash_attention_causal"', "wrapper name"
    )
    target.write_text(text)


def model_generator(source, target, core):
    text = source.read_text()
    text = replace_once(text, r'("block_size_m"\s*:\s*\()256(,\))', r"\g<1>64\2", "tile")
    text = replace_once(text, r'("is_causal"\s*:\s*\()0(,\))', r"\g<1>0, 1\2", "causal grid")
    text = replace_once(text, r"fa\.CausalVariant\.LOWER_RIGHT", "fa.CausalVariant.UPPER_LEFT", "mask")
    target.write_text(text)

    kernel = core / "src/infiniop/ops/flash_attention/ninetoothed/flash_attention.py"
    text = kernel.read_text()
    text = replace_once(
        text, r"range\(-\(-actual_kv_len // key\.dtype\.shape\[0\]\)\)",
        "range((actual_kv_len + key.dtype.shape[0] - 1) // key.dtype.shape[0])", "KV tail",
    )
    text = replace_once(
        text, r"(?ms)^([ \t]+)if is_causal:\n.*?(?=^\1next_max)",
        lambda match: (
            f"{match[1]}if is_causal:\n"
            f"{match[1]}    query_pos = query[i].offsets(-2)\n"
            f'{match[1]}    qk = ntl.where(query_pos[:, None] >= key_pos[None, :], qk, float("-inf"))\n\n'
        ),
        "causal mask block",
    )
    text = replace_once(
        text,
        r"query_i = \(1\.4426950408889634 \* scale \* query\[i\]\)\.to\(ntl\.float32\)",
        "query_i = query[i]",
        "BF16 query",
    )
    text = replace_once(
        text,
        r"qk = ntl\.dot\(query_i, ntl\.trans\(key\[j\]\)\.to\(ntl\.float32\)\)",
        "qk = ntl.dot(query_i, ntl.trans(key[j])) * (1.4426950408889634 * scale)",
        "BF16 QK with FP32 scale",
    )
    text = replace_once(
        text,
        r"ntl\.dot\(stable_qk, value\[j\]\.to\(ntl\.float32\)\)",
        "ntl.dot(stable_qk.to(value[j].dtype), value[j])",
        "BF16 PV with FP32 accumulator",
    )
    kernel.write_text(text)


def patch_dispatch(core):
    descriptor = core / "src/infiniop/ops/flash_attention/ninetoothed/descriptor.h"
    text = descriptor.read_text()
    text = replace_once(text, r'(block_size_m_\s*(?:=\s*|\{\s*))256\b', r"\g<1>64", "tile")
    text = replace_once(
        text, r'(#include "([^"]*/)flash_attention\.h")',
        r'\1\n#include "\2flash_attention_causal.h"', "causal include",
    )
    text = replace_once(
        text, r"(?m)^(\s*)if \(launch_flash_attention\(",
        r"\1const auto launch = _is_causal ? launch_flash_attention_causal : launch_flash_attention;"
        r"\n\1if (launch(", "causal dispatch",
    )
    descriptor.write_text(text)


def patch_launchers(core):
    guarded = 0
    guard = "#ifndef __HIP_PLATFORM_AMD__\n#define __HIP_PLATFORM_AMD__ 1\n#endif\n"
    generated = core / "build/ninetoothed"
    for path in generated.iterdir():
        if path.suffix not in {".c", ".cpp", ".h"}:
            continue
        text = path.read_text()
        def relocate_include(match):
            header = match[2]
            if not (generated / header).is_file():
                raise ValueError(f"Missing generated header {header} in {path.name}")
            return f'{match[1]}"{header}"'

        text = re.sub(
            r'(?m)^(\s*#\s*include\s+)"/[^"\n]*/build/ninetoothed/([^/"\n]+)"',
            relocate_include,
            text,
        )
        if not text.startswith(guard):
            text = guard + text
        if path.suffix == ".cpp" and "hipCtxGetId" in text:
            text = "#include <mutex>\n" + replace_once(
                text, r"hipCtxGetId\(NULL,\s*&ctx_id\);",
                "int launch_device = -1;\n"
                "    auto device_error = hipGetDevice(&launch_device);\n"
                "    if (device_error != hipSuccess) return device_error;\n"
                "    ctx_id = static_cast<decltype(ctx_id)>(launch_device);\n"
                "    static std::mutex module_mutex;\n"
                "    const std::lock_guard<std::mutex> guard(module_mutex);",
                f"device cache in {path.name}",
            )
            guarded += 1
        elif path.suffix == ".cpp" and "hipModuleLaunchKernel" in text:
            found = re.search(r"hipFunction_t\s+(\w+)_func\s*=\s*nullptr;", text)
            if not found:
                raise ValueError(f"unknown vendor launch cache in {path.name}")
            name = found[1]
            text = "#include <mutex>\n" + text
            text = replace_once(text, rf"\b{re.escape(name)}_func\s*=\s*nullptr;", f"{name}_func[64]{{}};", "function array")
            text = replace_once(text, rf"\b{re.escape(name)}_mod\s*=\s*nullptr;", f"{name}_mod[64]{{}};", "module array")
            text = replace_once(text, rf"void unload_{re.escape(name)}\(void\)\s*\{{", f"void unload_{name}(void) {{\n    int dev = 0;\n    HIP_CHECK(hipGetDevice(&dev));", "unload device")
            text = replace_once(
                text, rf"void load_{re.escape(name)}\(\)\s*\{{\s*int dev\s*=\s*0;",
                f"void load_{name}(int dev) {{", "load device",
            )
            text = replace_once(
                text, rf"if\s*\(\s*{re.escape(name)}_func\s*==\s*nullptr\)\s*load_{re.escape(name)}\(\);",
                "int dev = -1;\n    auto device_error = hipGetDevice(&dev);\n"
                "    if (device_error != hipSuccess) return device_error;\n"
                "    if (dev < 0 || dev >= 64) return hipErrorInvalidDevice;\n"
                "    static std::mutex module_mutex;\n"
                "    const std::lock_guard<std::mutex> guard(module_mutex);\n"
                f"    if ({name}_func == nullptr) load_{name}(dev);",
                "launcher device",
            )
            text = re.sub(rf"\b({re.escape(name)}_(?:func|mod))\b(?!\[)", r"\1[dev]", text)
            guarded += 1
        if path.suffix == ".cpp" and "hipModuleLaunchKernel" in text:
            # This host wrapper is compiled without HIP device builtins. gfx936
            # uses 64-lane waves, the same value used by the compiled kernel.
            text = re.sub(r"\bwarpSize\b", "64", text)
        path.write_text(text)
    if not guarded:
        raise ValueError("no Hygon launch cache found in generated code")
    print(f"Patched per-device launch caches for {guarded} generated variants", flush=True)


if __name__ == "__main__":
    mode, *args = sys.argv[1:]
    if mode == "generate-causal":
        causal_generator(*(Path(arg) for arg in args))
    elif mode == "generate-model":
        model_generator(*(Path(arg) for arg in args))
    elif mode == "patch-dispatch":
        patch_dispatch(Path(args[0]))
        patch_launchers(Path(args[0]))
    elif mode == "patch-launchers":
        patch_launchers(Path(args[0]))
    else:
        raise ValueError(mode)
