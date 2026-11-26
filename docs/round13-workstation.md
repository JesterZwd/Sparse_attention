# CoMRAD Round 13 工作站迁移

本次发布含六档前端对拍与综合收据、14 个完整解码步的同资源回放、Qwen2.5 短上下文对拍，以及后续 GPU 阶段所需脚本、种子、样本哈希和配置。Qwen3 64K、Qwen2.5 32K、FWE/CWE、LongBench v2 的新闭环和新增整步采集仍待运行。 [阶段结果](../results/round13_comrad/checkpoint.md)

## 1. 环境和权重

从 GitHub 克隆后在仓库根目录运行。原环境为 Linux x86-64、Python 3.12.3、PyTorch 2.5.1+cu121、Transformers 4.51.3、RTX 3090 和 NVIDIA 驱动 535.183.01。回放需要 C++17 编译器。GPU 冒烟会记录实际显存峰值。

```bash
git clone --filter=blob:none https://github.com/JesterZwd/Sparse_attention.git
cd Sparse_attention
python3.12 -m venv .venv
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
  -r requirements.lock.txt -r requirements-ruler.txt
python3 scripts/download_model.py
.venv/bin/python -m scripts.fetch_round13_weights
```

已下载的模型也可通过工作站间复制。保留 `models/Qwen3-4B-Instruct-2507/` 和 `models/Qwen2.5-7B-Instruct/` 的完整目录及各自的 `source_manifest.json`。两组权重由脚本按固定版本校验。Llama 条件实验当前只有权重可用性收据。 [Qwen3 文件清单](../configs/qwen3-4b-modelscope-files.json) · [Qwen2.5 权重收据](../results/round13_comrad/model_adaptation/weight_availability.json)

## 2. 恢复冻结输入

若从原工作站取得 `round13-inputs.tar.gz`，在仓库根目录解压并校验。该迁移包保留原始实验输入、RULER 语料及 Punkt 缓存。GitHub 提供生成器、来源和哈希。

```bash
tar -xzf /path/to/round13-inputs.tar.gz
.venv/bin/python -m scripts.prepare_round13_data verify-workspace
```

也可在干净克隆中按固定提交重建。语料来自 RULER 指定的上游 URL。若上游内容改变导致哈希不符，使用原工作站迁移包，保持冻结样本。

```bash
mkdir -p .cache/ruler-corpus .cache/ruler-upstream
git -C .cache/ruler-upstream init
git -C .cache/ruler-upstream fetch --depth=1 https://github.com/NVIDIA/RULER.git db43f92ff634bbce1c33d1fb15858aec4bba509a
git -C .cache/ruler-upstream checkout --detach FETCH_HEAD
cp scripts/download_ruler_corpus.py .cache/ruler-download-corpus.py
.venv/bin/python .cache/ruler-download-corpus.py
.venv/bin/python scripts/fetch_ruler_punkt.py

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1
.venv/bin/python -m scripts.prepare_round13_data workspace
```

使用迁移包时，只有 FWE 触发 CWE 回退才需要上述 RULER Git checkout。数据准备入口对照发布收据逐项检查测试行、顺序和哈希。重新生成的清单保留本机的创建时间和来源文件记录。 [数据口径与种子](../results/round13_comrad/data_protocol.md)

## 3. 接续 GPU 实验

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m scripts.queue_round13
```

队列依次执行 64K 直接对照、Qwen2.5 32K 拟合与校准、Qwen2.5 独立集、FWE/CWE、LongBench v2、32K/64K 新整步采集、回放和配对汇总。每题原子保存检查点，使用相同代码、输入、模型和运行配置重启可接着运行。队列状态和分阶段日志写入 `.cache/round13/`。逐题结果、冻结配置、置信区间和物理成本写入 `results/round13_comrad/`。手动逐阶段命令见执行方案。 [阶段脚本与命令](superpowers/plans/2026-11-25-round13-comrad.md)

主比较固定 P32、affine32 KV4、5% 物理读上限、0.80 V 预留和 64-bin → 64-bin 回退。分别比较 M 对 144 B A 和 M 对 Quest INT4，M 配额为 `min(C112, A144+4)`。新模型用 4 条拟合、16 条校准、64 条独立提示。提质验收线为 +1 pp、配对 95% 区间下界大于零、每步读量增加至多 1% 和零超限。保质省读线为区间下界至少 −1 pp、每步少读至少 5%、主算术计数下降和零超限。 [配额与门槛](../gqa_probe/round13_comrad.py)

## 4. CPU 回放与 RTL

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q
.venv/bin/python -m scripts.replay_round13_comrad --output .cache/round13/replay_check
```

默认回放使用已经发布的 14 个历史整步，自动核验并重组 Round 12 的 32K 分片。两种长度的新整步均完成后可加 `--include-new`。全套 RTL 对拍、同库综合和系数校准使用 Icarus Verilog 12.0、Yosys 0.33、ABC 1.01、OpenSTA 3.2.0 和指定 Nangate45 typical Liberty。仓库中的 12 个小型 RTL 向量只含量化输入记录、系数和期望分数。运行命令及 Liberty 哈希见硬件说明。 [硬件复现](../results/round13_comrad/hardware/README.md) · [回放结果](../results/round13_comrad/replay/)
