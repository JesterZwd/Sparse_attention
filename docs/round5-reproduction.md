# Round 5 共享补 K 实验复现

基线 `9cfae202c992e67ef20c511b84bf49a5086e4d18`。沿用固定 RULER v1 生成器、验签 Qwen 模型和项目 Python 环境。独立校准和测试的生成参数位于 `results/round5_shared_refill/prepare_calibration.json` 和 `prepare_test.json`，两份输出清单记录独立种子及最终 Qwen 输入哈希。复盘依赖工作站保留的第四轮 `results/round4_fixed_k_v/data/test`。所有长跑可从逐提示检查点恢复。

```bash
.venv/bin/python -m scripts.prepare_ruler --config results/round5_shared_refill/prepare_calibration.json --output results/round5_reproduction/data/calibration
.venv/bin/python -m scripts.prepare_ruler --config results/round5_shared_refill/prepare_test.json --output results/round5_reproduction/data/test
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u -m scripts.round5_pipeline --root results/round5_reproduction
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u -m scripts.round5_mechanism --output results/round5_reproduction/mechanism_verified
.venv/bin/python -m scripts.audit_round5_results --root results/round5_reproduction
```

准备命令使用空数据目录。校准汇总后冻结精确代码哈希、策略、读量匹配与比较，在 `frozen_test.json` 保存。测试从落盘的冻结文件读取策略和审计 ID。模型路径只有一个 affine32 前缀，后续每策略独立复制缓存。并发峰值统计逐层记录 8 个 KV 组的总有效候选页数。GPU 质量路径对选中 K 计算精确 logits，并以解量化 V 和 Mask 作参考。每步账本计逻辑实际候选的 INT4 码、scale/ZP、32 B burst 向上取整、索引和摘要。额外 K 读取的等待用首次 V 前的 K burst 数表示。审计脚本从逐提示结果及轨迹回执复核关键表。
