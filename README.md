# 嵩山杯赛题一：MiniMax-H3 FL2VA 八卡适配

本仓库提交参赛源码、InfiniCore 工作树覆盖文件、InfiniLM / NineToothed 源码以及构建脚本。可运行代码和详细说明位于 [`songshanbei/`](songshanbei/)。

## 目录说明

- [`songshanbei/src/`](songshanbei/src/)：八卡测试、九齿内核生成和 MHA 桥接实现
- [`songshanbei/overlays/`](songshanbei/overlays/)：InfiniCore 与 Python 兼容层覆盖文件
- [`songshanbei/vendor/`](songshanbei/vendor/)：InfiniLM、NineToothed 源码
- [`songshanbei/prepare.sh`](songshanbei/prepare.sh)：准备构建环境并编译依赖
- [`songshanbei/run.sh`](songshanbei/run.sh)：生成九齿内核、运行算子测试和 FL2VA
- [`songshanbei/README.md`](songshanbei/README.md)：环境要求、构建与运行步骤
- [`songshanbei/REPOSITORIES.md`](songshanbei/REPOSITORIES.md)：各组件的上游来源、分支和提交记录

## 快速开始

```bash
cd songshanbei
cp env.example.sh env.local.sh
# 编辑 env.local.sh，填写模型目录、首帧路径等本机路径
H3_ENV_FILE=./env.local.sh bash prepare.sh
H3_ENV_FILE=./env.local.sh bash run.sh
```

运行环境为 Linux x86_64、海光 gfx936 八卡、DTK 26.04、平台 PyTorch 2.9.0、Python 3.11、Git 和 xmake。模型权重和统一首帧使用主办方提供的文件，不包含在本仓库中。完整依赖和已知限制见 [`songshanbei/README.md`](songshanbei/README.md)。