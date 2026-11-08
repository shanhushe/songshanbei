#!/usr/bin/env bash

export H3_INFINICORE_SRC=/path/to/InfiniCore
export H3_INFINILM_SRC=/path/to/InfiniLM
export H3_INFINI_ROOT=/path/to/infinicore/install
# This directory contains ninetoothed/aot.py.
export H3_NINETOOTHED_PARENT=/path/to/ninetoothed/parent
export H3_VENDOR_WHEEL=/path/to/triton-3.5.1+das.opt1.dtk2604.torch290-cp311-cp311-manylinux_2_28_x86_64.whl
# This directory contains diffusers/ and the FL2VA checkpoint.
export H3_MODEL_PATH=/path/to/MiniMax-H3
export H3_FIRST_FRAME=/path/to/first_frame.jpg
export H3_OUTPUT_ROOT="$PWD/outputs"
# Optional. Use the environment's compatible Python dependencies, if required.
# export H3_RUNTIME_PYTHONPATH=/path/to/pylibs
# export H3_NT_BIN=/path/to/additional/ninetoothed/tools
