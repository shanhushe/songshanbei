# FL2VA 对比复现

使用 InfiniLM 原版 `examples/minimax_h3_compare.py` 比较两个已生成的 MP4。比较环境需安装 PyAV、NumPy、Pillow、scikit-image。

```bash
python /path/to/InfiniLM/examples/minimax_h3_compare.py \
  --tasks fl2va --model-path . \
  --reference-fl2va-video /path/to/reference_fl2va.mp4 \
  --infinilm-fl2va-video evidence/infinilm_fl2va.mp4
```

参考视频由 InfiniLM 的 `examples/minimax_h3.py fl2va --legacy` 使用同一权重、首帧、提示词、124 帧、50 个采样网格点和 seed 42 生成。平台单作业不足以完成 49 次去噪，因此此次使用 `resume_h3_reference.py` 在第 24 步保存视频与音频潜变量、采样器索引及随机数状态。第二个作业读取该保存点继续使用原去噪步骤。该脚本是参考生成的续跑辅助件，比较程序仍使用原版。

平台缺少 `skimage`，因此此次将参考视频下载至本地，核对两个文件的 SHA-256 后调用原版比较脚本。结果见 [comparison-result.txt](../evidence/comparison-result.txt)。参考视频的 SHA-256 为 `d4b00250061d05b1546c5c5b97d70c7fed6177f72edc32999f24650a86246d5c`。
