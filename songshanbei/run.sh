#!/usr/bin/env bash
set -euo pipefail

project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/src" && pwd)"
if [[ -n "${H3_ENV_FILE:-}" ]]; then
    source "$H3_ENV_FILE"
fi
: "${H3_INFINICORE_SRC:?Set H3_INFINICORE_SRC in env.local.sh}"
: "${H3_INFINILM_SRC:?Set H3_INFINILM_SRC in env.local.sh}"
: "${H3_INFINI_ROOT:?Set H3_INFINI_ROOT in env.local.sh}"
: "${H3_NINETOOTHED_PARENT:?Set H3_NINETOOTHED_PARENT in env.local.sh}"
: "${H3_VENDOR_WHEEL:?Set H3_VENDOR_WHEEL in env.local.sh}"
: "${H3_MODEL_PATH:?Set H3_MODEL_PATH in env.local.sh}"
: "${H3_FIRST_FRAME:?Set H3_FIRST_FRAME in env.local.sh}"
original=$(realpath "$H3_INFINICORE_SRC")
lm=$(realpath "$H3_INFINILM_SRC")
prefix=$(realpath "$H3_INFINI_ROOT")
model=$(realpath "$H3_MODEL_PATH")
first=$(realpath "$H3_FIRST_FRAME")
export H3_VENDOR_WHEEL=$(realpath "$H3_VENDOR_WHEEL")
nt_parent=$(realpath "$H3_NINETOOTHED_PARENT")
test -r "$nt_parent/ninetoothed/aot.py"
test -r "$H3_VENDOR_WHEEL"
test -r "$lm/examples/minimax_h3.py"
test -r "$first"
test -d "$model/diffusers"
test -r "$prefix/lib/libinfinicore_cpp_api.so"
if [[ "${1:-}" == --check-config ]]; then
    printf 'Configuration paths verified. No model was run.\n'
    exit 0
fi
output_root="${H3_OUTPUT_ROOT:-$PWD/outputs}"
mkdir -p "$output_root"
run=$(mktemp -d "$(realpath "$output_root")/model-XXXXXX")
chmod g+rx "$run"
printf 'H3_MODEL_RUN=%s\n' "$run"

{
scratch=$(mktemp -d "${TMPDIR:-/tmp}/h3-nine-model-XXXXXX")
core="$scratch/InfiniCore"
generated="$core/build/ninetoothed"
mkdir -p "$generated"
for pkg in src/infiniop/ninetoothed src/infiniop/ops/flash_attention/ninetoothed; do
    mkdir -p "$core/$pkg"
    cp -r "$original/$pkg/." "$core/$pkg/"
done
test -r "$first"
test -d "$model/diffusers"
test -r "$prefix/lib/libinfinicore_cpp_api.so"

if [[ -n "${H3_REUSE_MODEL_RUN:-}" && -s "$H3_REUSE_MODEL_RUN/generated.tar.gz" ]] &&
    gzip -t "$H3_REUSE_MODEL_RUN/generated.tar.gz"; then
    gzip -dc "$H3_REUSE_MODEL_RUN/generated.tar.gz" | tar -C "$generated" -xf -
    test -s "$generated/flash_attention.h"
    printf 'H3_REUSED_GENERATED %s\n' "$H3_REUSE_MODEL_RUN"
else
python3 -S "$project/prepare_nine_model.py" generate-model \
    "$project/gen_one.py" "$scratch/gen_model.py" "$core"
cp "$scratch/gen_model.py" "$run/"
cp "$core/src/infiniop/ops/flash_attention/ninetoothed/flash_attention.py" "$run/"
cd "$core"
printf 'H3_MODEL_AOT_START\n'
mkdir -p "$scratch/vendor-ninetoothed" "$scratch/vendor-bin"
tar -C "$nt_parent" -cf - ninetoothed | tar -C "$scratch/vendor-ninetoothed" -xf -
cp "$project/vendor_aot_bootstrap.py" "$scratch/vendor-ninetoothed/sitecustomize.py"
cp "$project/vendor_private_cache.py" "$scratch/vendor-ninetoothed/"
python3 -S - "$H3_VENDOR_WHEEL" "$scratch/vendor-ninetoothed" <<'PY'
import hashlib
import sys
from pathlib import Path
from zipfile import ZipFile

wheel, stage = map(Path, sys.argv[1:])
with wheel.open("rb") as source:
    digest = hashlib.file_digest(source, "sha256").hexdigest()
assert digest == "e397b16365da22508353ce44a45f5be82f8ee58e49853a3df06aa8f6f83d5581", digest
with ZipFile(wheel) as package:
    package.extractall(stage)
aot = stage / "ninetoothed/aot.py"
source = aot.read_text()
source = source.replace(
    'c_source_file_name = f"{kernel_name}.{signature_hash}.c"',
    'c_source_file_name = f"{kernel_name}.{signature_hash}.cpp"',
)
aot.write_text(source)
print("H3_VENDOR_WHEEL_VERIFIED", digest, flush=True)
PY
ln -s "$(command -v python3)" "$scratch/vendor-bin/python"
NINETOOTHED_ARCH=gfx936 ROCM_PATH=/opt/dtk-26.04 \
    TRITON_HIP_CLANG_PATH=/opt/dtk/llvm/bin/clang PATH="$scratch/vendor-bin${H3_NT_BIN:+:$H3_NT_BIN}:$PATH" \
    PYTHONPATH="$scratch/vendor-ninetoothed:$core/src" TRITON_CACHE_DIR="$scratch/triton-cache" \
    TRITON_CACHE_MANAGER=vendor_private_cache:PrivateCache \
    python3 -u -c 'import runpy, sys, triton; print("H3_VENDOR_TRITON", triton.__version__, triton.__file__, flush=True); runpy.run_path(sys.argv[1], run_name="__main__")' "$scratch/gen_model.py"
printf 'H3_MODEL_AOT_DONE\n'
tar -C "$generated" -cf - . | gzip -c > "$run/generated.tar.gz"
fi
python3 -S "$project/prepare_nine_model.py" patch-launchers "$core"

cd "$generated"
shopt -s nullglob
sources=(flash_attention_*.cpp)
[[ "${#sources[@]}" -gt 0 ]]
compiler=/opt/dtk/llvm/bin/clang++
"$compiler" -shared -fPIC -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ \
    -I"$original/include" -I/opt/dtk/include -Wno-return-type -Wno-unused-result \
    -Wl,-z,defs -x c++ flash_attention.c "${sources[@]}" \
    -o "$generated/libh3_nine_model.so" -L/opt/dtk/lib -lamdhip64
"$compiler" -shared -fPIC -O2 -std=c++20 -DH3_DIRECT_AOT \
    -I"$original/include" -I"$generated" -I"$project" \
    "$project/mha_nine_bridge.cpp" -L"$prefix/lib" -linfinicore_cpp_api \
    -L"$generated" -lh3_nine_model -Wl,-z,defs '-Wl,-rpath,$ORIGIN' \
    -o "$generated/libh3_nine_bridge.so"
printf 'H3_MODEL_BRIDGE_BUILT\n'
for library in libh3_nine_model.so libh3_nine_bridge.so; do
    cat "$generated/$library" > "$run/$library"
    sha256sum "$run/$library"
done

export INFINI_ROOT="$prefix"
export LD_LIBRARY_PATH="$generated:$prefix/lib:/opt/dtk/lib:/opt/dtk/cuda/cuda/lib64:/usr/local/lib/python3.11/site-packages/torch/lib:/opt/hyhal/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$project${H3_RUNTIME_PYTHONPATH:+:$H3_RUNTIME_PYTHONPATH}:$original/python:$lm/python${PYTHONPATH:+:$PYTHONPATH}"
export H3_NINE_MHA_BRIDGE="$generated/libh3_nine_bridge.so"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python3 -u "$project/test_nine_aot.py" "$generated/libh3_nine_model.so"
python3 -u "$project/test_nine_aot.py" "$generated/libh3_nine_model.so" --causal
python3 -u "$project/flash_attention_cases.py" --mha

cd "$lm"
printf 'H3_FL2VA_START\n'
python3 -u "$project/launch_nine.py" "$lm/examples/minimax_h3.py" fl2va \
    --model-path="$model" --first-frame="$first" --text="A dog falls asleep" \
    --gpus=0,1,2,3,4,5,6,7 --num-frames=124 --steps=50 --seed=42 \
    --output="$run/infinilm_fl2va.mp4" 2>&1 | tee "$run/generation.log"
grep -o 'H3_NINE_MHA_DISPATCH device=[0-7]' "$run/generation.log" |
    sort -u > "$run/verified-devices.txt"
test "$(wc -l < "$run/verified-devices.txt")" -eq 8
test -s "$run/infinilm_fl2va.mp4"
sha256sum "$run/infinilm_fl2va.mp4"
printf 'H3_FL2VA_NINE_8DCU_GENERATED\n'

} 2>&1 | tee "$run/model.log"
