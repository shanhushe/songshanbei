#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ -n "${H3_ENV_FILE:-}" ]]; then source "$H3_ENV_FILE"; fi
state=${H3_BUILD_ROOT:-$root/.h3-build}
mkdir -p "$state"
state=$(realpath "$state")
core="$state/InfiniCore"
lm="$state/InfiniLM"
deps="$state/pylibs"
export INFINI_ROOT=${H3_INFINI_ROOT:-$state/install}
export DTK_ROOT=${DTK_ROOT:-/opt/dtk}
if [[ -r "$DTK_ROOT/env.sh" ]]; then
    set +u
    source "$DTK_ROOT/env.sh"
    set -u
fi
export PATH="$DTK_ROOT/cuda/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$INFINI_ROOT/lib:$DTK_ROOT/lib:$DTK_ROOT/cuda/cuda/lib64:/usr/local/lib/python3.11/site-packages/torch/lib:/opt/hyhal/lib:${LD_LIBRARY_PATH:-}"
export CPATH="/usr/local/include/python3.11:${CPATH:-}"
export LIBRARY_PATH="/usr/local/lib:${LIBRARY_PATH:-}"
command -v xmake >/dev/null || { echo '需要安装 xmake 3.0.5 或兼容版本'; exit 1; }
command -v git >/dev/null
python3 -c 'import sys, torch; assert sys.version_info[:2] == (3, 11), sys.version; print(torch.__version__)'
if [[ ! -d "$core/.git" ]]; then
    git clone --no-checkout https://github.com/InfiniTensor/InfiniCore.git "$core"
fi
git -C "$core" checkout --detach f32e2fbb2a9b2ca912b5711f4f3836c403196a74
git -C "$core" submodule update --init --recursive -- third_party/spdlog submodules/InfiniRT submodules/InfiniOps submodules/InfiniCCL
cp -a "$root/overlays/InfiniCore/." "$core/"
mkdir -p "$lm"
cp -a "$root/vendor/InfiniLM/." "$lm/"
mkdir -p "$deps" "$INFINI_ROOT"
python3 -m pip install --target "$deps" -r "$root/requirements-h3.txt"
cp -a "$root/overlays/python/." "$deps/"
export PYTHONPATH="$deps:$core/python:$lm/python${PYTHONPATH:+:$PYTHONPATH}"
if [[ -d "$lm/third_party/pybind11/include" && ! -d "$core/third_party/pybind11/include" ]]; then
    mkdir -p "$core/third_party/pybind11"
    cp -a "$lm/third_party/pybind11/include" "$core/third_party/pybind11/"
fi
jobs=${H3_BUILD_JOBS:-16}
(
    cd "$core"
    xmake f -y -cv --hygon-dcu=y --hygon-arch=gfx936 --cuda="$DTK_ROOT/cuda/cuda" --ccl=y --aten=y
    xmake build -j "$jobs"
    xmake install
    xmake build -j "$jobs" _infinicore
    xmake install _infinicore
)
(
    cd "$lm"
    xmake f -y -cv -m release
    xmake build -j "$jobs" _infinilm
    xmake install _infinilm
)
