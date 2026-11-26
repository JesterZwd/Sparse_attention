# CoMRAD · Llama-3.1-8B 六档预算验证

在 Llama-3.1-8B-Instruct 上使用 affine32 KV4、P32、0.80 V 预留和 64-bin 安全前缀回退。32K 和 64K 各用 64 条独立 RULER Multi-Value 提示。一个 Llama 掩码在 4 条 32K 拟合提示上生成，并冻结给两个长度共用。各长度另有 16 条量化校准提示，Dense BF16 和 Dense KV4 均得 100%。

| 上下文 | 物理读上限 | INT4 Quest | 144 B A | 112 B M | M 读量／解码步 |
|---|---:|---:|---:|---:|---:|
| 32K | 5% | 48.83% | 67.58% | 74.61% | 51.77 MB |
| 32K | 7% | 79.30% | 88.28% | 89.45% | 70.73 MB |
| 32K | 10% | 92.19% | 94.53% | 94.14% | 99.32 MB |
| 64K | 5% | 54.69% | 67.97% | 71.09% | 100.61 MB |
| 64K | 7% | 82.03% | 80.47% | 82.81% | 139.27 MB |
| 64K | 10% | 88.67% | 90.23% | 89.84% | 197.45 MB |

32K 和 64K 的 Dense KV4 独立分分别为 99.61% 和 95.31%。5% 下，M 对 Quest 的配对增益在 32K 为 +25.78 pp [+16.80, +34.77] pp，在 64K 为 +16.41 pp [+9.38, +23.44] pp，每解码步物理读量分别下降 6.81% 和 9.45%。相对 A，32K 通过质量门槛，64K 每步少读 11.71%，但配对区间下界 −1.95 pp 低于 −1 pp 非劣门槛。区间为配对 95% bootstrap。

复核了 160 份逐题包、1,344 条记录和两个长度对同一冻结掩码的引用。11,995,904 个独立集稀疏组步零超限。

[完整质量和成本表](quality_cost.csv) · [32K 配对](32k/test/paired.json) · [64K 配对](64k/test/paired.json) · [掩码](channels.json) · [验收日志](reproduction/accept-llama-matrix.log) · [文件哈希](autodl_artifacts.json)

## 从固定版本复现

运行环境为 Linux、Python 3.12.3、PyTorch 2.5.1+cu124、Transformers 4.51.3 和 NVIDIA A800 80 GB。安装项目依赖时使用对应 PyTorch CUDA wheel。按 [Round 13 数据准备](../../docs/round13-workstation.md#2-恢复冻结输入)准备 RULER 固定提交、语料正文和 Punkt。以下命令从仓库根目录运行，`R13_PYTHON` 可指定已经配置好的 Python。

```bash
export R13_PYTHON="${R13_PYTHON:-.venv/bin/python}"
export PYTHONPATH="$PWD" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
base=results/round13_llama_budgets
rep="$base/reproduction"
cache=.cache/round13-cloud/llama32k
mkdir -p "$cache"
cp "$rep/official-tree.json" "$rep/mirror-tree.json" "$cache/"
"$R13_PYTHON" "$rep/llama32k-fetch.py"
"$R13_PYTHON" "$rep/prepare_llama_ruler.py" --config "$rep/prepare_pilot.json"

export COMRAD_LLAMA_OUTPUT=.cache/round13-llama-rerun
for length in 32k 64k; do
  "$R13_PYTHON" "$rep/prepare_llama_ruler.py" \
    --config "$base/prepare_$length.json" --output "$COMRAD_LLAMA_OUTPUT/data/$length"
done

"$R13_PYTHON" - <<'PY'
import json
import os
from pathlib import Path
from results.round13_llama_budgets import run_matrix

root = Path(os.environ["COMRAD_LLAMA_OUTPUT"])
published = Path("results/round13_llama_budgets/reproduction")
for length in ("32k", "64k"):
    actual = json.loads((root / "data" / length / "manifest.json").read_text())
    expected = json.loads((published / f"{length}_manifest.json").read_text())
    assert actual["splits"] == expected["splits"], length
run_matrix.ROOT = root
run_matrix.MASK = root / "channels.json"
run_matrix.main()
PY
```

[固定模型收据](reproduction/model_source_manifest.json)记录镜像 SHA256 与官方 Git LFS 对象的对应关系。发布的配置仍保留 AutoDL 原运行中的路径和运行时标识。重建写入单独目录，新产物可与发布的逐题结果和哈希配对复核。AutoDL 的后台与终端监视脚本保留在此目录。公开逐题表保存分数、读量、生成长度和输出 token 哈希。原始提示、权重和续跑包保存在原执行环境。
