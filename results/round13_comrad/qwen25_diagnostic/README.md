# Qwen2.5 · 32K Dense 量化复测

原 16 条 Multi-Value 的 Dense BF16 为 93.75%，affine32 KV4 为 0%。在其中两题上，只量化 K 到 4-bit 时均降到 0，仅量化 V 时均保留 BF16 分数。

| 校准面板 | BF16 | affine32 KV4 | K8/V4，32 维分组 |
|---|---:|---:|---:|
| 32K Multi-Value，16 题 | 93.75% | 0% | 93.75% |
| 32K FWE，4 题 | 75.00% | 0% | 75.00% |

K8/V4 与 BF16 逐题同分。K4 改为 16 维分组后，FWE 四题仍为 0，Multi-Value 也在两题上复现。诊断只使用 Dense 路径和校准样本。它支持把 K8/V4 作为候选量化配置。若用于稀疏配置，需要按新的 K 页字节重新核算物理预算。

[Round 13 原量化对照](../qwen25_32k_multivalue/calibration/quantization.json) · [K/V 隔离诊断](qwen25-acceptance-diagnostic.json) · [FWE 和 K 位宽结果](results.json) · [16 题 K8/V4](k8-full-mv-calibration.json)

脚本沿用 Round 13 工作区。先恢复原 Qwen2.5 Multi-Value 数据，再在仓库根目录执行以下命令。数据生成环境包含 SciPy 1.15.3。

```bash
export R13_PYTHON="${R13_PYTHON:-.venv/bin/python}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 PYTHONPATH="$PWD"
d=results/round13_comrad/qwen25_diagnostic
cache=.cache/round13-cloud/qwen25-alternate-task
mkdir -p "$cache"
cp "$d/config.json" "$cache/config.json"
"$R13_PYTHON" -m scripts.prepare_ruler --config "$cache/config.json"
"$R13_PYTHON" "$d/qwen25-acceptance-diagnostic.py"
"$R13_PYTHON" "$d/qwen25-alternate-task.py"
"$R13_PYTHON" "$d/qwen25-k8-full-calibration.py"
```
