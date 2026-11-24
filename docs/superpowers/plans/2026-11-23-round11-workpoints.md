# Round 11 · 112 B 摘要工作点

1. 沿用 Round 10 的 Q8 评分、P32、affine32、固定 96 维掩码和 hist64 安全前缀。A 按 144 B 公式分配，B 用自身 112 B 账本并受 A 的页数限制，C 以自身预算分配。V 预留统一为 .80。用尾页、保护、回退、页数和冻结门槛的回归验收分配器。
2. 32K 新校准 16 条，同时采共同状态和自由生成。A/B 在共同状态中的保护、实际 K 记录数和 QK 必须相同。每提示保存官方分和读取、计算计数。周期样本包括 6 个代表层、答案值首 token、回退压力组以及 3 个完整 decode step。
3. 若 B 的配对校准区间下界低于 −1 pp，或 C 的后端 QK+PV 每步增加超过 10%，补一个 M 档。M 的 K 页数为 min(C, A+4)。只有 M 校准官方分至少提高 1 pp 且单位步后端 MAC 少于 C 时，才和 A/B/C 一并进入独立验证。
4. 校准完成后封存 32K 的 128 条新提示 ID、控制策略、输入哈希和门槛。B 走用户指定的非劣、省读、主算术判定。C/M 走用户指定的提质判定。C/M 硬件判定另检查 64-lane 主情景的均值、P95、容量和能耗。
5. 用独立 LPDDR 回放和能耗分析统一同一批地址、同一资源、同一算术账。32/64/128 lane 都跑，8 周期/页与 96 维 6 周期/页分别标注。容量不可行点保留阶段需求。能耗按不同参数情景、按答实际长度估计。
6. 将 32K 入选工作点与 A 在 Round 10 的 64K 校准配对，做同目录回放。先检查质量、预算与 SRAM/P95，再用 Round 10 未使用的后 64 条提示冻结验证。小样本跨任务从 32K VT 开始。两条主策略同时饱和时，再跑已有的 16K FWE。80 B 等 112 B 主线完成后择机。
7. 对照 Round 10 的同方法回归、检查冻结哈希和完整性，交付报告、逐题表、分阶段回放与能耗盈亏平衡表。数据和大中间文件留在工作站。

## 复现命令

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python -m scripts.prepare_round11
for phase in check32 regression cal32 common32 test32 cross cal64; do
  .venv/bin/python -m scripts.run_round11 "$phase"
done
.venv/bin/python -m scripts.replay_round11 --states results/round11_workpoints/32k/common/states.jsonl.gz --output results/round11_workpoints/32k/replay
.venv/bin/python -m scripts.finish_round11 audit32
```

有 64K 入选配置时，按 `64k/common/states.jsonl.gz` 回放至 `64k/replay`，再运行 `scripts.finish_round11 freeze64`、`scripts.run_round11 test64`、`scripts.finish_round11 audit64`。能耗脚本用闭环计数和共同状态回放，按每条路线实际解码步数推算每答案能耗。

在 64K 共同状态回放恢复时，可用 `scripts.run_round11 common64` 读取已完成的校准入围记录。64K 校准按 −1 pp 点差筛查可用的任务质量，独立集仍按各工作点指定的区间和成本判定。若 VT 平均分至少 95% 且逐题分数全同，`scripts.finish_round11 fwe` 补做低区分度情形的 FWE 检查。
