# 固定 K 的 V 跳读复现

代码基于 `4fd60ca`。使用第二、三轮的 Qwen 模型、校验文件、RULER 固定生成器与语料。测试版本、数据哈希和样本清单保存在 `results/round4_fixed_k_v/{diagnostic,32k_mv,test}/config.json`。正式 held-out 配置是 `frozen_test.json`，包含工作点、阈值、冻结数据清单哈希与校准回执。

三个阶段由 `scripts.round4_pipeline` 调度。已有结果目录可以按 prompt 检查点恢复，相同配置与代码哈希为恢复条件。换目录复现时，先用 `scripts.prepare_ruler` 读取该轮两份 `prepare_*.json`，指定新目录中的 `data/32k_mv` 与 `data/16k_fwe` 作为输出。诊断点使用第三轮保留在工作站的 `results/round3_summary_bytes/data/16k`。

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u -m scripts.round4_pipeline --root results/round4_fixed_k_v
```

32 B burst 模型的 K/V 页面使用分段对齐的连续码、Scale 和 Zero Point 平面。索引流按连续页 ID 列表计费，摘要流另计。码使用逐向量 FP16 Scale，Quest 向外舍入。统计的 K/V/摘要是参考布局事务量。地址回放有固定的开放行映射，输出的是 first-V 依赖的 K burst、事务数和 row hit。GPU 使用解量化缓存与 Mask PV 作质量参照。

主流程生成 `report/`。只有出现已诊断的尖峰漏选后才单独运行 `scripts.round4_dual_probe`。曲线脚本 `scripts.plot_round4_curve` 使用 Matplotlib 3.10.8，从已生成的校准 CSV 导出 SVG 和 PNG。
