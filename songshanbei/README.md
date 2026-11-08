# MiniMax H3 FL2VA 八卡适配

嵩山杯初赛，单人队（李云静）。本项目在八张海光 gfx936 DCU 上把九齿生成的注意力接入 InfiniLM 的 FL2VA 推理。验证版输出 768×928、124 帧、24 fps 的音视频，模型入口耗时 1203.970 秒，与 PyTorch 参考视频的平均 SSIM 为 45.293%。

## 仓库内容

- `src/` 包含八卡验证包的九个原始文件及 `gen_one.py`。`prepare_nine_model.py` 在临时副本中应用算子改动，包含 64 行 tile、KV 尾块、因果掩码、BF16 矩阵运算和启动器兼容修正。`mha_nine_bridge.cpp` 与 `mha_nine_run.inc` 负责接入 InfiniCore MHA。
- `run.sh` 是从验证版启动脚本整理出的可配置入口。它生成并构建九齿算子，执行数值测试，运行 FL2VA，然后核对八张卡的九齿调用记录。
- `env.example.sh` 是路径配置示例。`evaluation/` 记录此次参考对比的方法。`evidence/` 包含八卡视频、对比程序的结果段和作业编号。

## 环境和版本

八卡实测使用 InfiniFabric 的 `scnet-dcu` 环境 `524820f26b72`，Python 3.11、PyTorch 2.9.0 海光构建、DTK 26.04。九齿编译使用 Triton 3.5.1 `das.opt1.dtk2604.torch290` 的 CPython 3.11 wheel。运行端还需要与 H3 兼容的 Diffusers、Transformers、Accelerate、Safetensors 和视频编码依赖，单独进行视频对比需要 PyAV、NumPy、Pillow 和 scikit-image。

| 依赖 | 来源 | 复现状态 |
| --- | --- | --- |
| InfiniLM | [InfiniTensor/InfiniLM](https://github.com/InfiniTensor/InfiniLM) | 实测使用平台上的源码与编译扩展，具体 commit 和本地改动待核对 |
| InfiniCore | [InfiniTensor/InfiniCore](https://github.com/InfiniTensor/InfiniCore) | 需要启用海光 DCU、ATen 和九齿相关选项，具体 commit 和本地改动待核对 |
| NineToothed | [InfiniTensor/ninetoothed](https://github.com/InfiniTensor/ninetoothed) | 实测使用平台上的兼容版本，具体 commit 和已有 `aot.py` 改动待核对 |
| Triton wheel | [海光 DTK 软件源](https://pypi.sourcefind.cn/das_release/dtk2604-rc1/+simple/triton/) | `triton-3.5.1+das.opt1.dtk2604.torch290-cp311-cp311-manylinux_2_28_x86_64.whl`，SHA-256 `e397b16365da22508353ce44a45f5be82f8ee58e49853a3df06aa8f6f83d5581` |
| MiniMax-H3 Base FL2VA | [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) | 使用主办方提供的权重目录和统一首帧 |

平台原环境还含 `pylibs` 兼容层，具体文件及版本待核对。此包保留了已验证的适配源码和输出，但新整理的入口尚未在另一套干净八卡环境完整跑过。

## 构建和运行

1. 按上述上游项目的构建说明，在海光 DTK 环境编译并安装兼容版本。InfiniCore 的安装前缀应含 `lib/libinfinicore_cpp_api.so`，InfiniLM 源码目录的 Python 扩展需已构建并可导入。已有竞赛环境可以指向其相应目录。
2. 把 `env.example.sh` 复制为 `env.local.sh`，填写源码、安装前缀、wheel、模型和首帧路径。兼容包如有需要，通过 `H3_RUNTIME_PYTHONPATH` 加入搜索路径。
3. 在仓库根目录执行：

   ```bash
   H3_ENV_FILE=./env.local.sh bash run.sh --check-config
   H3_ENV_FILE=./env.local.sh bash run.sh
   ```

默认从源码生成九齿产物。脚本先测试因果和非因果注意力及高层 MHA，再使用统一提示词 `A dog falls asleep`、124 帧、50 个采样网格点、seed 42 生成视频。输出目录的 `generation.log`、`verified-devices.txt` 和 `infinilm_fl2va.mp4` 可用于检查结果。干净构建和推理的总时间可能超过竞赛平台单次 3480 秒限制。已有环境如需分开编译和运行，可以用 `H3_REUSE_MODEL_RUN` 指向自己从同一版本源码生成、包含 `generated.tar.gz` 的目录。

仓库不分发固定模型权重、主办方输入、第三方 wheel、编译库、缓存或保存点。
