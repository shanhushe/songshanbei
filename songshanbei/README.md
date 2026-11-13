# MiniMax-H3 八卡源码补充版

单人队的 FL2VA 九齿注意力适配。此目录补充 InfiniCore 的工作树覆盖文件、InfiniLM 和 NineToothed 实际源码、Python 兼容层以及构建脚本。`src/` 保留八卡 v8 测试和 MHA 桥接实现。

## 构建与运行

使用 Linux x86_64、海光 gfx936 八卡、DTK 26.04、平台 PyTorch 2.9.0 海光版、Python 3.11、Git 和 xmake。构建需要能够访问 GitHub 和 Python 软件源，Triton wheel 可从海光软件源读取或在配置中指定。模型权重和统一首帧使用主办方提供的文件。

```bash
cp env.example.sh env.local.sh
# 编辑 env.local.sh，填模型目录和首帧路径。
H3_ENV_FILE=./env.local.sh bash prepare.sh
H3_ENV_FILE=./env.local.sh bash run.sh
```

`prepare.sh` 从固定的 InfiniCore 基线构建，叠加恢复的覆盖文件，随后构建附带的 InfiniLM。`run.sh` 生成九齿内核并编译桥接库，跑算子用例和八卡 FL2VA。平台单作业若有时长上限，可先在具有编译工具的工作区运行 `prepare.sh`，将 `H3_BUILD_ROOT` 设在跨作业可读写的目录。运行脚本会写出生成日志、每设备调用记录和 MP4。

原 v8 平台记录和本地比较结果见 `evidence/`，比较方法见 `evaluation/`。本次补包未进行新的八卡复跑。InfiniCore 覆盖来自成功取回的部分工作树，平台完整差异未全部取回，因此尚不能保证与当时的已安装版本完全一致。

提交信息见 [仓库与版本](REPOSITORIES.md)。
