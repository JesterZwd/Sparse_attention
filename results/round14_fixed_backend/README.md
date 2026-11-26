# CoMRAD Round 14 · 定点后端校准与同域回放

本轮在 AutoDL A800 80 GB 上比较 112 B M 的浮点后端参考与两种定点后端候选。四个校准面板沿用冻结的模型、掩码、5% 物理读上限、0.80 V 预留、保护页和 `min(C112, A144+4)` 配额。共 52 条校准提示、156 条逐路线记录。 [逐题数据](output/) · [冻结配置与来源](publication.json)

## 定点闭环

候选接入 Light-A1 页评分归一化、INT8 Query 与 affine32 K4 的整数 QK、带误差界的定点 64-bin 和带 V scale/ZP 的 PV。`UINT8 Block-Max` 逐头逐页缩放，`UINT16` 逐头采用统一尺度。两条路线同时改变 P 的编码位宽和缩放。模型投影、稠密 prefill 和模型其余路径沿用 BF16。 [定点参考](reproduction/fixed_backend.py)

| 校准面板 | 提示数 | 浮点后端 M | UINT8 Block-Max | UINT16 |
|---|---:|---:|---:|---:|
| Qwen3 32K FWE | 8 | 91.67% | 91.67% | 91.67% |
| Qwen3 32K Multi-Value | 16 | 92.19% | 93.75% | 95.31% |
| Qwen3 64K Multi-Value | 12 | 95.83% | 93.75% | 93.75% |
| Llama 3.1 32K Multi-Value | 16 | 79.69% | 75.00% | 82.81% |

校准通过线为官方分较同批参考下降至多 2 pp、组步零超限、logits 零饱和。两条路线在 Qwen3 64K 均下降 2.08 pp。Llama 32K 的 UINT8 下降 4.69 pp，UINT16 提高 3.13 pp，后者配对 95% 区间为 [−4.69, +10.94] pp。UINT16 优先作为后端候选，位宽冻结需要后续归因和硬件验证。八个定点配置合计 1,259,232 个组步，超限与 logits 饱和均为零。 [校准汇总与配对区间](calibration_summary.csv)

280 个向量通过独立 NumPy CPU 的逐位复算，涵盖 QK、Light-A1、64-bin、PV 和物理读量账本。实测校准轨迹未出现自然回退，定向回退测试覆盖了账本、保护页和初读 K 的费用。发布的逐题结果保留指标及生成 token 哈希，原始 prompt、模型权重和数值激励留存在本地归档。 [向量哈希和复算收据](vector_acceptance.json) · [定向测试](reproduction/test_fixed_backend.py)

## 同域时钟回放

沿用 Round 11–13 的 24 个 Qwen3 完整步，32K 为 13 步，64K 为 11 步。A/M 共同使用相同前后端逻辑时钟时，仅去除评分器结果侧的跨时钟域延迟，保留内存接口 CDC 和缓冲、信用设置。下表为 M 相对 A 的平均 Attention 回放时延变化。

| 时钟配置 | 32K | 64K |
|---|---:|---:|
| Round 13 分域参照 | −3.14% | −6.66% |
| 共同 1.80 ns | −1.12% | −4.96% |
| 共同 2.00 ns | −1.13% | −5.09% |

两种共同周期下，A 和 M 的绝对平均时延均高于原分域投影。此处的时序是地址轨迹回放，完整引擎的可达频率待 RTL 和布线后 STA 确认。 [同域汇总](clock/clock_summary.csv) · [24 步逐步记录](clock/clock_step_timing.csv) · [时序来源与哈希](clock/receipt.json)

## 复算与重跑

从仓库根目录复算已发布的 12 行校准结果，环境需要 Python 3.12 和 NumPy。

```bash
round14=results/round14_fixed_backend
python "$round14/reproduction/report_calibration.py" "$round14/output"
```

GPU 重跑沿用 [Round 13 模型及 RULER 环境](../../docs/round13-workstation.md) 和 [Llama 模型、冻结掩码及数据](../round13_llama_budgets/README.md)。准备各面板的校准行，核对其 `calibration.jsonl` 的 SHA256 与对应 `output/<panel>/reference/config.json` 中的 `rows_sha256` 一致。

| 面板 | 生成配置 | 运行时输入目录 |
|---|---|---|
| Qwen3 32K FWE | [Round 13 FWE](../round13_comrad/prepare_qwen3_32k_fwe.json) | `results/round13_comrad/data/qwen3_32k_fwe` |
| Qwen3 32K Multi-Value | [Round 11 的 16 条校准](reproduction/qwen3_32k_mv_prepare.json) | `.cache/comrad-round14/data/qwen3_32k_mv` |
| Qwen3 64K Multi-Value | [Round 10 64K 校准](../round10_summary_64k/prepare_64k.json) | `.cache/round13-cloud/old_64k_calibration` |
| Llama 3.1 32K Multi-Value | [Llama 32K 配置](../round13_llama_budgets/prepare_32k.json) | `results/round13_llama_budgets/data/32k` |

在仓库根目录启动缓存目录里的脚本，以便输出沿用原实验目录结构。另开终端可运行 `bash .cache/comrad-round14/monitor.sh`。

```bash
round14=results/round14_fixed_backend
run=.cache/comrad-round14
mkdir -p "$run"
cp "$round14/reproduction/"* "$run/"
export PYTHONPATH="$PWD" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python "$run/test_fixed_backend.py"
for panel in qwen3_32k qwen3_64k llama_32k; do
  for route in reference uint8 uint16; do
    python -u "$run/run_calibration.py" "$route" "$panel"
  done
done
for route in reference uint8 uint16; do
  python -u "$run/run_qwen3_32k_mv.py" "$route" qwen3_32k_mv
done
python "$run/verify_vectors.py" "$run/output" --receipt "$run/vector_receipt.json"
python "$run/report_calibration.py" "$run/output"
python "$run/replay_clock.py"
```
