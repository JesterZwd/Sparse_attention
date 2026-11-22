# Round 7 预算与 V 门控实验

基础提交为 `4f0dd8a7d2b1f5ab5a29627f6c0218916db16c0c`。本轮复用 Round 6 的 16 条 32K Multi-Value 校准和此前未解码的 128 条独立提示。数据由对应的 `prepare_calibration.json` 和 `prepare_test.json` 重新生成，模型和数据校验与 Round 6 相同。运行前记录校准和测试的哈希，运行后保留逐提示评分与物理读账。

```bash
.venv/bin/python -m scripts.prepare_ruler --config results/round6_subpage_head/prepare_calibration.json --output results/round6_subpage_head/data/calibration
.venv/bin/python -m scripts.prepare_ruler --config results/round6_subpage_head/prepare_test.json --output results/round6_subpage_head/data/test
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python -u -m scripts.round7_pipeline
```

GPU 显存不足时可用 `--device cpu --threads 4 --root <新目录>`。权重和 KV 数值格式沿用 BF16 与 affine32。CPU 与 CUDA 的结果在各自后端内配对，报告的周期来自依赖回放模型。代码与配置校验在同一结果目录约束了后端。冒烟入口 `scripts.round7_run smoke` 使用独立结果目录，覆盖单遍回退、直方图、紧凑摘要和两项支线。

## 数值与预算

单遍 K 的候选上限包含保护页。下发 K 前为保护页 V 预留完整 burst。超限时仅从已读 K 的排序前缀重算 V 掩码，首次得到的掩码尚未发出 V 请求。成本仍计入初始全部 K 和索引、检测及重算。5% 点固定相同字节上限，分别报全读 V、在全读额度上多取四页，以及 K 预留允许的最大页数。128 条独立集用于校准通过后的冻结对照。

逐组步文件每条记录为 `[decode_step, layer, ceiling_bytes, groups]`。每组字段依次为总读字节、已读 K 页、已读 K token、V 并集 token、回退标志、有效 K 前缀页数、回退前超限字节、逐头保留 token 数之和。逐组步以实际物理布局逐项验收。闭环时序沿用 Round 6 层和步抽样。共同轨迹另采样生成正确值首 token 前的 Query。组步 BF16 数值参考在软件做掩码后的矩阵乘，MAC 账反映映射后有效工作量。峰值 logits 账使用实际有效 token 数，软件填充与运行时分配另计。

## LPDDR5 与选择器

复用 Round 6 的 32 B burst、地址映射、LPDDR5-6400 x16 模型和逐阶段依赖屏障。此处按模型 CK 计费。单遍计摘要/索引、K、门控和预算检查、V、PV。PV 固定阵列按共享物理保留 token 数发射，另列各头实际有效 MAC。摘要更新写量与读量分列。动态回放将新增索引 burst 放在补 K 阶段，重算扩大分母所需的初始 logits 和掩码 SRAM 读也纳入该阶段。

64-bin 采用 0.25 自然对数间隔，最后一桶饱和。使用 FP32 质量，强制保护后整桶保留临界桶。周期和状态账假设每组每 CK 读四个分数、每头一套直方图，并复用全局峰值归约。该实现对应 [SPARTA 的阈值结构](https://arxiv.org/html/2610.34824v1#S4)，用本项目的 GQA 参考数值与存储布局。动态补 K 路线沿用精确 Top-p 闭环，单独标注基于 64-bin 的分析成本代理。精确 Top-p 提供数值对照。

同一候选必须符合零超限、官方分数 95% 配对区间下界至少 −1 个百分点，以及总阶段模型周期或控制 SRAM 降幅至少 5%。若所有配对得分均相同，百分位 bootstrap 区间会退化，报告会保留这一局限。小模块综合、流式候选、64K 及模型扩围由本轮结果触发。

主流程结束后运行 `.venv/bin/python -m scripts.run_round7_cold_estimate`，按冷页真值 P99 触发四提示的近似冷质量闭环。使用 `.venv/bin/python -m scripts.audit_round7_results` 验证逐组步读量、冻结清单和收据。
