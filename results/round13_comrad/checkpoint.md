# Round 13 · CoMRAD 泛化与同边界前端

论文工作题目为 **CoMRAD — Compact Page Summaries for Sparse Attention in Edge LLM Decoding**。

方法名为 **CoMRAD**，展开为 Contribution-Pruned Mean–Radius Attention Decoding。保留 112 B M 的初选 K 配额 `min(C112, A144+4)`。同任务比较按 P32、5% 物理上限、.80 V 预留、保护页、A1 与 64-bin 安全前缀回退运行。实际读量由各路径的地址账本单列。Quest INT4 + 64-bin 采用 SPARTA 式算法对照口径。SPARTA 的系统包含三值权重、统一数据通路与专用流控。

[SPARTA 原论文](https://arxiv.org/abs/2610.34824) · [方案与复现](../../docs/superpowers/plans/2026-11-25-round13-comrad.md)

## 数据封存

Round 12 未开封的 64 条 Qwen3 64K Multi-Value 与 25 个历史小组实例哈希零重叠。Qwen2.5 使用分离的 4/16/64 条拟合、校准、独立提示。FWE 使用 8/64 条，CWE 已准备独立回退种子。LongBench v2 以模板后实际长度筛 64 条，覆盖六个官方领域，范围为 16,466–31,848 个输入 token，另留 128 个生成 token。

[64K 封存](qwen3_64k_seal.json) · [RULER 样本](ruler_data_receipt.json) · [LongBench 封存](longbench_receipt.json) · [数据复现](data_protocol.md)

## 模型适配

Qwen2.5-7B-Instruct 使用官方 BF16 快照的四个分片，全部通过固定版本 SHA256 校验。真实权重读取到 28 层、28 Query 头、4 KV 头、128 维、原生 32,768 总 token 上限、RoPE θ=1,000,000。模型的实际 Q/K 归一化层数为零。CPU 短上下文 37 token 的适配对拍中，原生与接入路径的贪心续写一致。分块 Prefill logits 相对 RMSE 为 3.535%，分组单步与原生的相对 RMSE 为 1.489%。独立的 7:1 算法回归检查了真实头数的 QK 算术与 5% 上限。32K GPU 验收另外测分块 BF16/KV4 的 allocated 和 reserved 峰值。

Llama-3.1-8B-Instruct 官方模型需要访问授权。当前匿名解析返回 HTTP 401，工作站暂未取得权重。

[模型来源](model_adaptation/weight_availability.json) · [Qwen2.5 短上下文对拍](model_adaptation/qwen25/short_parity.json)

## 同边界前端

两种宽度共用真实 512 B 接收 FIFO、每拍双发系数、相同宽乘加尾部与 Nangate45 约束。六档通过真实记录、换 Query、随机/长期反压和标签对拍。4/3 在两种宽度下的映射持续服务最快。112 B 的 0.5 拍/记录残余气泡较小，保留字节齐备后的统一发射方案。

| 摘要 | 槽 / 信用 | 平均拍/记录 | 临界周期 ns | ns/记录 | 映射面积 µm² |
|---|---:|---:|---:|---:|---:|
| 144 B A | 2/2 | 15.0 | 1.66351 | 24.953 | 245,308 |
| 144 B A | 4/2 | 8.5 | 1.65794 | 14.093 | 274,329 |
| 144 B A | 4/3 | 8.0 | 1.75119 | 14.009 | 274,710 |
| 112 B M | 2/2 | 13.0 | 1.66961 | 21.705 | 228,999 |
| 112 B M | 4/2 | 7.5 | 1.76599 | 13.245 | 246,633 |
| 112 B M | 4/3 | 6.5 | 1.74542 | 11.345 | 248,653 |

4/3 的 112 B 记录服务时间和面积相对 144 B 分别下降 **19.02%** 和 **9.49%**。六种映射的临界周期均高于 1.25 ns。786 条连续记录的接收、首次 dot、释放与结果事件和回放模型逐项对拍。

[数值与反压](hardware/rtl_validation.json) · [同库综合](hardware/synthesis.json) · [回填校准](hardware/frontend_calibration.json) · [硬件口径](hardware/README.md)

## 复用轨迹的首次同资源回填

重跑 Round 11 的 6 个整步及 Round 12 新采的 8 个整步，各工作点用自己的完整前端映射周期。共享 64 lane、1.25 ns 后端、64 KiB 阶段 SRAM 和同样接收 FIFO。表取 4/3。单位为每整步毫秒。后端组在一条控制器上执行，CDC 与端口按同一假设计。

| 轨迹来源 | 长度 | 整步数 | A 均值 | B 均值 | M 均值 | M P95 | M 峰值 SRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| Round 11 | 32K | 3 | 10.574 | 9.558 | 10.246 | 10.417 | 39,754 B |
| Round 12 | 32K | 5 | 10.488 | 9.479 | 10.155 | 10.260 | 39,604 B |
| Round 11 | 64K | 3 | 20.271 | 18.240 | 18.934 | 19.251 | 52,240 B |
| Round 12 | 64K | 3 | 20.234 | 18.195 | 18.884 | 19.030 | 51,991 B |

两批现有轨迹上，32K 的 M 时延改善约 3.1%–3.2%，64K 约 6.6%–6.7%。4/2 与 4/3 的 M 整步时延几乎持平。持续输入下的较小服务间隔在实际请求返回与后端重叠下只有部分收益。

[回填收据](replay/replay_receipt.json) · [逐步时序](replay/step_timing.csv) · [时序配对](replay/paired_timing.csv) · [能耗情景](replay/paired_energy.csv)

## GPU 执行状态

工作站只有一张 RTX 3090。首次尝试恰逢共享 GPU 的批处理切换子任务，下一子任务重新占用显存时，Qwen3 在加载权重阶段因显存不足退出。该尝试尚未写入测试配置或生成逐题结果。后台队列等待占用 GPU 的整批父进程退出，并确认显存空闲后启动。队列保留逐阶段日志，单个面板失败时继续独立面板。

Qwen3 64K、Qwen2.5 32K、FWE/CWE 和 LongBench v2 的闭环分数，以及本轮新增完整步回放仍待执行。跨面板汇总将校验冻结配置和逐题收据。现有回填只覆盖前两轮已采样的 14 个完整步。

[CPU 验证收据](cpu_verification.json) · [工作站迁移](../../docs/round13-workstation.md) · [后台队列](../../scripts/queue_round13.py) · [跨面板汇总](../../scripts/report_round13_comrad.py)
