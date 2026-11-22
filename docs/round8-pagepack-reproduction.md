# Round 8 页内 K/V 压缩与四档 V 门控

基础提交为 `da8f8985bf6ce164a1a6afe891518985f8cc54f8`。32K Multi-Value 使用 Round 7 校准集和相同的 Qwen3-4B-Instruct-2507、affine32、Sink/Local 与 P32。数据和模型按 [Round 7 复现说明](round7-reproduction.md) 准备。前 8 条采集共同前缀，前后各 4 条用于固定页数的压缩过筛，16 条用于 5% 校准闭环。在评分前记录脚本与数据摘要。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python -u -m scripts.probe_pagepack capture
.venv/bin/python -m scripts.probe_pagepack report
.venv/bin/python -m scripts.screen_range_gate results/round8_pagepack/probe/b/*.jsonl.gz --output results/round8_pagepack/b_summary.json
.venv/bin/python -m scripts.audit_pagepack_probe
.venv/bin/python -u -m scripts.run_pagepack
```

`traces/` 是本地真实原始 QK 记录。它同时保存 A1 和 B 的候选页 ID 与答案值首 token 位置。报告以同提示分位数作汇总，原始提示和模型权重另存。

## A1 与 A2

两 token 的 K 统计逐级组合成四 token 与整页的 sum/min/max/count。整页摘要始终源于原始 RoPE 后 K；压缩模式共用 5% 下原始 Mean6+Radius2 全读 V 的候选页 ID。联合压缩使用段 `K_mean/V_mean` 和段长修正 `log n`。只压 V 时 softmax 仍保留逐 token 的 K 质量并按 V 段求和。protected 页及尾页仍为逐 token 记录，原始 singleton 保持其数值。新代表先用 FP32 均值定位损失，再按 affine32 的 INT4 code、FP16 scale、UINT8 ZP 编码后反量化到模型 dtype。

混合段长按 `||max-min||₂/||mean||₂` 决定合并。只用 K 门槛和 K+V 各自检查门槛都测 .5、.8 和 1.2。A1 的相对 Dense 绝对 P99 在一般状态与答案首 token 状态都至多增加 5%，相对原始选页输出的 P99 不超过 .35，且合计 K/V burst 至少少 10%。两半校准同时满足才进入压缩闭环。混合目录与段标签另计，复算表另外给出仅扣 16 B/页目录的剩余节省上界。

A2 每步按同一公式选择最长预算前缀。全读 V 按实长全额预留，64-bin 对未保护 V 预留完整字节的 65%，保护页全额预留。V 并集超过预算后缩到可全读 V 的最长前缀，再用相同 64-bin 复算掩码，已读 K 全部计费。固定 stride 直接寻址。可变 stride 加 16 B/页目录和独立对齐的 2-bit 段长标签。闭环是数值仿真，burst 由布局推导并与抽样地址回放核对。有效 MAC 与实际 GPU 指令分列含义见 Round 7。

`group_steps.jsonl.gz` 内，每步记录 `[decode_step, layer, cap_bytes, groups]`。每组字段依次为读取字节、已读 K 页、K 代表数、共享 V 代表数、回退标记、回退后前缀页数、目录字节、标签字节。时序以抽样组报告。双缓冲仅报 K 与 QK 理想重叠的周期下界并计两个满页缓冲，精确回退仅比较 token 槽 ID 位数和 16-bit 槽开销。

## B

固定原始 INT6 Quest 候选 K，比较 `64-bin → full`、`64-bin → 64-bin`、四档严格 2% 和四档 2→3→4%。按每头最小合法档生成 V 掩码，再求共享并集。放宽后复用质量累计，变化时重新生成 mask 和核对物理 burst。失败走相同安全前缀的全读 V，累计质量分别记在已读 K 与最终前缀分母下。完整时序沿用 bank 并发模型，容量是已定义缓冲的峰值同时驻留量。

闭环结束后运行 `.venv/bin/python -m scripts.audit_pagepack_results` 复核逐组步读量。实验结果见 [Round 8 报告](../results/round8_pagepack/report/checkpoint.md)。
