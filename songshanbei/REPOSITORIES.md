# 初赛报告需列出的代码

以下目录应由同一个公开提交固定。将 `songshanbei/` 内容上传至 `shanhushe/songshanbei` 的 `master` 分支后，把 GitHub 显示的完整新 commit 填进同一份初赛报告。若老师要求独立仓库链接，则把三个目录分别发布到公开仓库，并填写各仓库实际分支和提交号。

| 组件 | 公开提交中的路径 | 上游来源和版本 |
| --- | --- | --- |
| 项目入口、桥接与内核生成 | `songshanbei/src/`、`songshanbei/run.sh`、`songshanbei/prepare.sh` | [提交仓库](https://github.com/shanhushe/songshanbei)，分支 `master`，新 commit 上传后填写 |
| InfiniCore | `songshanbei/overlays/InfiniCore/` | [平台记录的上游基线](https://github.com/InfiniTensor/InfiniCore/commit/f32e2fbb2a9b2ca912b5711f4f3836c403196a74)，工作树覆盖随新提交 |
| InfiniLM | `songshanbei/vendor/InfiniLM/` | [InfiniTensor/InfiniLM](https://github.com/InfiniTensor/InfiniLM)，平台目录无 Git 元数据，附该快照的编译源码 |
| NineToothed | `songshanbei/vendor/ninetoothed/` | [InfiniTensor/ninetoothed v0.26.0](https://github.com/InfiniTensor/ninetoothed/releases/tag/v0.26.0)，含平台安装版本的 Python 源码 |
| Python 兼容层 | `songshanbei/overlays/python/` | 依赖版本见 `requirements-h3.txt`，当前恢复四个文件 |

初赛报告写入公开链接、branch 和此次提交的完整 SHA。上游基线的 SHA 与你的 GitHub 提交 SHA 分开记录。
