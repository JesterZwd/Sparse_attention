# Round 12 · 窄摘要与评分器前端

- 从 Round 10 四条拟合提示复算全维贡献，核对冻结 96 维。分别重建 80、64 维的摘要码和尺度，沿用全维 Q8 量化。
- 在 12 条 Round 11 校准提示上重放同一批 Q/K/V，复用 112 B M 初选 K 配额。对拍 K 工作量，按首值证据保留与 P99 过筛。把可用候选送入自由生成，优先 96 B。只冻结一个最强候选；其 C 档只作校准诊断。
- 32K 新独立集 128 条。64K 先用原 12 条校准验证，若可用，再开新 64 条。独立门槛为配对区间下界至少 −1 pp、每步少读至少 5%、零超限。两个长度分别报告。
- 用可见的真模型记录与定向向量核对 RTL 输入、系数和 Q24 分数。计算核、首取数、前端接页间隔、单/双发和可达时钟分开记录。原 128 核、96 预置核、96 完整基准与依据关键路径的流水变体使用同库约束。
- 重跑 Round 11 原 3 步及不同提示的追加整步，用实际接页间隔、时钟、端口与槽数回填共享 LPDDR5 模型。按阶段列时序、峰值工作集和能耗敏感性。保存配置和账本并同步 GitHub main。

## 复现

GPU 阶段依赖已有的 Round 10 拟合和 Round 11 校准数据。调用 `scripts.prepare_ruler --config` 及本轮 `prepare_32k.json`、`prepare_64k.json` 生成新测试数据。运行如下。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for phase in data fit capture32 cal32 capture64 cal64 cal_c test32 test64; do
  .venv/bin/python -m scripts.run_round12 "$phase"
done
```

RTL 和综合命令见 [硬件复现](../../results/round12_frontend/hardware/README.md)。整步回放从两轮收据读入配对状态。用本轮两份综合和 RTL 收据识别 112 B 与 96 B 的可达时钟及服务周期。

```bash
r11=results/round11_workpoints
r12=results/round12_frontend
if [ ! -f "$r12/32k/extended/states.jsonl.gz" ]; then
  cat "$r12"/32k/extended/states.jsonl.gz.part* > "$r12/32k/extended/states.jsonl.gz.tmp"
  printf '%s  %s\n' \
    c4bcf26bbc4e0bbd68eecedf9dc2102f50e9395d40d001e93cd40d85deca8f35 \
    "$r12/32k/extended/states.jsonl.gz.tmp" | sha256sum -c - &&
    mv "$r12/32k/extended/states.jsonl.gz.tmp" "$r12/32k/extended/states.jsonl.gz"
fi
.venv/bin/python -m scripts.replay_round12 \
  --states "archived=$r11/32k/common/states.jsonl.gz" \
  --states "archived=$r11/64k/common/states.jsonl.gz" \
  --states "extended=$r12/32k/extended/states.jsonl.gz" \
  --states "extended=$r12/64k/extended/states.jsonl.gz" \
  --verify-archive "archived/32k=$r11/32k/replay/step_timing.csv" \
  --verify-archive "archived/64k=$r11/64k/replay/step_timing.csv" \
  --rtl-report "$r12/hardware/rtl_validation.json" \
  --mapping-report "$r12/hardware/synthesis.json" \
  --narrow-rtl-report "$r12/hardware/narrow_rtl_validation.json" \
  --narrow-mapping-report "$r12/hardware/narrow_synthesis.json" \
  --output "$r12/replay"
```
