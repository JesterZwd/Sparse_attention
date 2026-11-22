# Round 7 硬件复核

基线是 `a8db79b7e81526f0fa2281b7b85fdcb20e36ea7f`。复用 Round 7 的前 8 条 32K Multi-Value 校准提示、层和解码抽样。Dense KV4 生成共同前缀，采集时核对原有 Dense 的得分、完整命中和解码步数。统一验证所有 `(sample_id, layer, step, group)` 与已发布固定 K 记录相同。复核的输出误差均相对同状态的 Dense KV4 输出。

## 运行

先按 [Round 7 复现步骤](round7-reproduction.md) 准备模型和校准数据。

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python -u -m scripts.replay_round7_review capture --histogram-fallback
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u -m scripts.replay_round7_review replay
.venv/bin/python -m scripts.replay_round7_review summary
```

输出目录为 `results/round7_hardware_review`。`addresses/` 保存逐组物理 burst 地址和前置阶段等待，`timing/` 保存对应命令时序、峰值工作集和质量诊断。两套目录各有 SHA-256 清单。每条路检验整个地址集合无重复 burst，读字节逐项与预算账一致。与输入内容有关的 Q/K/V 和原始提示留在数值路径内。

## 路径与计费

- 固定 INT6 Quest 排名、P32、10% 读上限和最多 68 个 K 页。单遍门控为 64-bin 或无直方图 `τ=8`。同一门控分别使用精确 Top-p、安全前缀全读 V 和安全前缀 64-bin 回退。保护页、前缀外概率损失和全部已读 K 均按实际情况计费。
- 动态路线首批保守地为全 V 留额度，补页前为新增 K/V 和索引预留字节。两次都用相同的 64-bin 定义，补 K 后锁定首批掩码，按扩大后的分母和已丢质量决定新增 token。记录重读首批 logits、掩码和预算校验。与单遍的比较涉及候选深度和最终掩码。
- 片上直接寻址使用 `base + page_id × stride`，计入选中页 ID 等片上状态。片外页表按物理 Page ID 读 4 B 表项所在的 32 B burst，初次索引在 Scorer 和 Top-k 完成后发出。补页复用首批已读表项所在 burst，缓存计入工作集，补读在预算决策后发出。两种布局分别重算预算。
- 保留 Round 6 的 LPDDR5-6400 x16 参数和地址映射，使用 32 请求窗口、bank 内 FIFO、跨 bank 就绪调度。阶段顺序为摘要、首批索引、首批 K、追加索引、追加 K、V。控制器在阶段间等待读数据和计算依赖。另在两层、三步抽样把窗口降为 1，以同地址检查并发敏感性。模型省略刷新、写事务、WCK 切换和组间并发。首 V 可发时间、首命令和首 RD 分列。
- SRAM 按各阶段同时存活的缓冲取峰值。分别报可复用的 Scorer 分数、门控状态、有效 logits 和 PV 状态。Dense 使用同样的 logits 和 PV 格式。模型工作集覆盖这些已定义缓冲。共用的 Q/K/V 暂存、请求队列和流水寄存器需在 RTL 实例中定容。

复核结论见 [报告](../results/round7_hardware_review/checkpoint.md)。周期与读取的配对区间先在每条提示内平均，再使用 10,000 次配对 bootstrap。采集的源文件快照和实际使用的 bank 回放版本分别记录在两阶段收据里，后者包含 nPPD 校验。
