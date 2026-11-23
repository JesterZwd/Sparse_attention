# Round 9 Mean6+Radius2 融合评分器

基线为 `5a3a27ab84981aa0e1580d43f85140ea21ebfb1d`。保持 P32、原始 affine32 KV、单遍 K、64-bin 和安全前缀 64-bin 回退。Round 8 历史校验使用原提交的代码校验和。本轮也把独立质量与物理成本判定接回历史实验运行器。

## 数值和硬件

Q8 每头对称量化，scale 跨该头的页复用。两个尺度系数以 20-bit unsigned Q24 入模块，分别吸收 Q8 scale、摘要 scale 和 Attention 缩放，Radius 同时吸收 `α=0.25`。MR 的独立 `log n` bias 为 signed Q24。Quest 因最小值、最大值使用独立尺度，按 query 符号分别累加再缩放。A1 保持原有 FP32 归一化、求和和 Page ID 并列规则。评分 RTL 的公共输入端接收已换算的系数和 bias。换算和 A1 的面积未进入模块综合，换算周期在总账作敏感性分析。

`rtl/score_datapaths.sv` 比较 Quest、同通道数的串行与融合 MR，以及真正每 8 CK 接页的双串行。16 通道每拍，128 通道每页。等吞吐串行共享 Query 缓冲和公共后端。融合半径项用两位门控、移位与相加来累积，abs Query 在装载时计算。用采集的 Q 和摘要对照 RTL 输出整数，VCD 切换计数描述活动水平。

## 运行步骤

按 Round 8 固定模型和 RULER 生成数据。评分器校准会在本地 `local_inputs/` 保留用于数值复核的 Q 和摘要。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python -u -m scripts.run_fused_scorer calibrate
.venv/bin/python -m scripts.audit_scorer_numerics --inputs results/round9_fused_scorer/local_inputs \
  --output results/round9_fused_scorer/hardware --rtl-fixtures .cache/round9/fixtures
PATH="$PWD/.cache/round9/tools/bin:$PATH" .venv/bin/python tests/test_score_rtl.py \
  --quest-vectors .cache/round9/fixtures/quest_vectors.json \
  --mr-vectors .cache/round9/fixtures/mr_vectors.json \
  --vcd-dir .cache/round9/vcd --report results/round9_fused_scorer/hardware/rtl_validation.json
PATH="$PWD/.cache/round9/tools/bin:$PATH" .venv/bin/python -m scripts.synthesize_score \
  --result results/round9_fused_scorer/hardware/synthesis.json --activity-dir .cache/round9/vcd
.venv/bin/python -u -m scripts.run_fused_scorer independent
.venv/bin/python -m scripts.report_fused_scorer
```

先用四条校准提示在 5% 和 7% 搜索预留系数，按实际字节匹配并靠近相同 .65 预留的平均位置。剩余 12 条验证匹配。正式集只运行验证匹配的点。在 7% 有匹配时加 INT6 的 10% 单遍基线。五组独立运行的判定先检查零超限，质量提升路线要求每解码 token 实际读量相差不超过 1%、官方分至少 +1 pp、95% 配对区间下界大于零。非劣省读路线要求质量区间下界至少 −1 pp、每答案读量降幅的区间下界至少 10%。

系统时序从 LPDDR 模型的串行依赖周期扣去原评分点积估算，再按共同 STA 可行时钟插入小模块测得的页接收间隔、流水延迟和 Query 装载。四个头按复制的小模块并行。系数供给另报每页 0、2、8 个模块 CK 的敏感性。LPDDR 时钟采用 1.25 ns。内存地址、原有 A1 周期和后续 64-bin、QK/PV 阶段沿用原模型。

## 同库综合和门级活动

本次采用 Yosys 0.33、ABC 1.01、Icarus 12.0 和 OpenSTA 3.2.0。Liberty 为 Nangate45 typical，SHA-256 为 `8d540a4d4cf6d09d27c87ad067857a9c0c2eeb023ab7a56e058cd3113db4e9b1`。Yosys、ABC 和 Icarus 用 Ubuntu 软件包安装或本地解包，OpenSTA 源码构建。可用工具也可以通过 `--yosys`、`--abc` 和 `--sta` 显式指定。Liberty 可用 `--lib` 指定。

同批真实先导页用 22-bit 系数重新编码为位级向量。门级切换结果的流程是先 `--mapping-only` 输出映射收据，再以 Yosys `read_liberty -ignore_miss_func` 和 `write_verilog -noattr` 生成仿真标准单元。逐顶层从收据目录取 `mapped.v`，和单元 Verilog 一起传给 `tests/test_score_rtl.py --top … --rtl … --rtl … --vcd-dir …`。最后用 `--reuse-mapping`、`--activity-dir` 与 `--activity-stage mapped_gate` 运行综合/STA 汇总。映射约束为 1.25 ns 目标、BUF_X1 输入驱动、5 fF 输出负载。切换计数来自功能门级仿真，未做布线和 SDF 回注。

闭环完成后运行 `.venv/bin/python -m scripts.audit_fused_scorer_results`。GitHub 结果保留逐提示预算、给出实际地址布局的核算代码和抽样时序。完整逐组步压缩账本及 bundle 由工作站作业生成，原文件 SHA-256 在验证收据中便于复核。原始 Q 和摘要、波形、工具二进制和权重保存在本地。
