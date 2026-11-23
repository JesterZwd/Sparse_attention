# Round 10 紧预算算法与 64K 验证

1. 用新种子生成 16K 对拍、32K 校准和新测试、64K 校准和新测试。按实例哈希检查两个长度的测试与历史样本隔离。32K 独立集 128 条，64K 预先选前 64 条。
2. 分别实现 P32/P64 物理读量、112 B 两种摘要及分块 Prefill。旧 P32 与 Round 9 对照验收，P64 支持尾页、保护扩张、直址 burst 与安全前缀。
3. 32K 前四条拟合原始 K 幅值与真实评分贡献的层/组共享掩码，剩余校准回放。固定 24 P32 页及相近的 12 P64 页，另给 Oracle 44 页。按共同证据阶段分清 K 召回和 V 保留。
4. 用同一 5% 硬上限比较所有候选的回放质量与字节。只让一个合格结构进入校准闭环。匹配实际字节后冻结并在新 128 条上验证。64 维、层预算与温度仅在有证据时补做。
5. 16K/32K 对拍整段和 2048 分块，再执行 64K 显存测试及 5% Dense、Quest、旧 MR 校准。若两条稀疏都低于 90% 或都距 Dense 超过 8 pp，增加 7%。先在各长度冻结选择，最后才进入独立集。

## 判定口径

- 32K 的结构筛选按提示汇总。相对原 Mean6+Radius2，答案值首 token 的证据 token 保留率最多降 1 pp，P99 绝对输出误差最多升 5%。在此前提下，证据保留率至少升 1 pp 且读量最多升 1%，或读量至少降 5% 且 K、PV 工作量都不增加，才进入闭环校准。
- 质量主比较要求实际读量相差不超过 1%，官方分提高至少 1 pp，逐提示配对 95% 区间下界高于零。省读比较要求区间下界达到 −1 pp 非劣界，每答案读量至少降低 10% 且其区间也支持该门槛，QK 和 PV 有效 MAC 都降低。两条路径均要求各组步零超限。
- 32K 首值诊断用 Dense 同状态轨迹。K 召回按已读 K 计算，另给回退后前缀召回。V 同时报告保留证据所在页和实际证据 token。P32 固定 24 页与 P64 固定 12 页的比较按实际有效 token 数核对。
- 每 Decode Token 读量按总布局读量除以完成的 decode 步数。每答案读量保留生成长度差异。GPU 路径保存仿真的 affine32 反量化缓存，读 burst 和有效 MAC 来自布局与掩码账本。

## 运行

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python -m scripts.run_round10 data
.venv/bin/python -m scripts.run_round10 diagnostic
.venv/bin/python -m scripts.run_round10 closed32
.venv/bin/python -m scripts.run_round10 closed64
.venv/bin/python -m scripts.run_round10 test32
.venv/bin/python -m scripts.run_round10 test64
.venv/bin/python -m scripts.audit_round10_pages
```

数据按照结果目录内的 `prepare_16k.json`、`prepare_32k.json` 和 `prepare_64k.json` 用 `scripts.prepare_ruler` 生成。闭环脚本按提示原子写入，复跑时核对源代码、模型、数据和策略配置。两个闭环校准依次冻结各自的工作点，再执行独立比较。页 ID 审计重放冻结的 32K 共同状态，核对物理读量和证据命中后保存入选 K 与保留 V 的页 ID。
