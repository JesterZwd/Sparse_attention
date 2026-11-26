# Round 13 · CoMRAD 泛化与同边界前端

- 封存 Round 12 的新 64K Multi-Value 样本，并为 Qwen2.5、FWE/CWE 和 LongBench v2 核对来源、重叠与实际 token 数。M 保留 112 B 和 A+4，Quest INT4 和 A 使用 144 B。所有稀疏路线共用 P32、5% 上限、.80 预留和 64-bin 回退。
- GPU 空闲后运行 Qwen3 64K 主表。Qwen2.5 先做短上下文数值与 32K 显存验收。4 条训练掩码、16 条量化/稀疏校准、64 条新种子测试彼此隔离。Llama 在权重和显存条件具备时启动。
- FWE 小校准先检查逐题差异。若仅一条或更少的校准提示能区分各稀疏方法，按预案转 CWE。LongBench v2 用固定自然长文题和官方选项解析。每个面板分别计算 M 对 A、Quest 的置信区间，报告实际读量和算术代理。
- 相同 RTL 边界验证 144/112 B 的 2/2、4/2 与 4/3，计入 512 B 接收 FIFO，区分系数、记录槽、结果信用的压力。实测约束选择配置，再把两边可达时钟回填相同 LPDDR 和后端模型，扩充完整步来源。
- 结果按任务独立报告。7:1 只报告算法和真实七头工作量；前端硬件结论使用四头设计。完整引擎决策参考新模型、新任务和同边界回填数据。

## 运行

先按 [工作站迁移说明](../../round13-workstation.md) 准备文件。GPU 阶段依次运行如下。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python -m scripts.run_round13 qwen3_64k
.venv/bin/python -m scripts.check_round13_model --stages smoke
for phase in qwen25_fit qwen25_cal qwen25_test fwe_screen; do
  .venv/bin/python -m scripts.run_round13 "$phase"
done
# 当 task_selection.json 选中 CWE 时，先生成该批样本。
# .venv/bin/python -m scripts.prepare_ruler --config results/round13_comrad/prepare_qwen3_32k_cwe.json
# .venv/bin/python -m scripts.prepare_round13_data verify-ruler
.venv/bin/python -m scripts.run_round13 fwe_test
.venv/bin/python -m scripts.run_round13 longbench
for length in 32k 64k; do
  .venv/bin/python -m scripts.capture_round13_comrad "$length"
done
.venv/bin/python -m scripts.replay_round13_comrad --include-new
.venv/bin/python -m scripts.report_round13_comrad
```

[硬件口径](../../../results/round13_comrad/hardware/README.md) 含六档对拍、综合和校准命令。省略 `--include-new` 时回填已验证的 Round 11/12 共 14 个整步。
