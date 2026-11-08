# 八卡实测证据

- [InfiniLM 八卡生成作业](https://tools.infinitensor.com/compute/scheduler/#/batch/88e541c9a9834834ba248af929ee0e34)的日志记录了设备 0–7 的九齿 MHA 调用，生成 MP4 后在参考对比阶段达到平台时限。
- [参考生成第一段](https://tools.infinitensor.com/compute/scheduler/#/batch/5ac578e478c943288a77ed87ddfbfb33)正常退出并保存第 24 步状态。[参考生成第二段](https://tools.infinitensor.com/compute/scheduler/#/batch/dfcec50e55a34f4aa81df3e1f9601415)恢复状态并生成参考 MP4，平台对比阶段因缺少 `skimage` 退出。
- [比较脚本结果](comparison-result.txt)来自在本地对两个已有 MP4 运行原版 InfiniLM 比较程序。视频尺寸和帧数均通过，平均 SSIM 为 45.293%。
- [InfiniLM 输出样例](infinilm_fl2va.mp4)的 SHA-256 为 `8a5c61cebc404ae00862c3f6c220a6602560ed987c934da987a8892838ba9013`。
