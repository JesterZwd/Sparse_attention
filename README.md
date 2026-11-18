# GQA Joint Page Selection — Minimal Probe

依据 `端侧LLM_GQA联合TopKTopP_研究与实验方案_V1.1.md`，首轮实现 A1、A3、全页候选 A3-B。研究原始方案保持不变。

## 已实现的实验口径

- 固定物理 Page 预算，默认 Page=32，Sink=1 Page，Local=128 Tokens，所有保护页去重并计费。
- 预算 `B=floor(J*fraction)`，5%/10%/20%；非法预算直接报错，不偷偷提高预算。
- 覆盖目标 0.7/0.8/0.9，A3-B 批大小 1/4/8/16。排序均为得分降序、Page ID 升序。
- 动态阶段全部达标后按预建 A1 列表补页；补页不计入改变率或动态轮次。
- 每个 Head 在完整有效历史上独立归一化；Sparse 输出直接从选中原始 Logits 重算稳定 Softmax。
- 明细包含 Head 覆盖/误差、批内冗余、实际批大小、改变率、单 Head 必要可行性上界和选择器工作量。
- P5/P95/P99 对全部记录的 Query Heads 汇总，不平均单组百分位。样本、层、步之间有关联，记录数量不能当成独立样本数。

## 环境

Python 已由系统提供；依赖在项目 `.venv`，无需更改系统 CUDA 或驱动。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements.txt
.venv/bin/python -m pytest -q
```

本工作站使用驱动 535.183.01、RTX 3090，已验证 PyTorch 2.5.1+cu121 的 BF16 GPU 运算。受限沙箱不暴露 GPU，采集命令需在工作站有 GPU 访问权限的终端运行。

若本机代理超时，可使用安装时验证过的高校镜像：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY .venv/bin/python -m pip install --cache-dir .cache/pip --index-url https://pypi.tuna.tsinghua.edu.cn/simple --find-links https://mirrors.nju.edu.cn/pytorch/whl/cu121/torch/ --timeout 30 --retries 1 -r requirements.txt
```

## 模型与 Trace 采集

默认模型是 `Qwen/Qwen3-4B-Instruct-2507`。当前工作站通过 Qwen 官方 [ModelScope 仓库](https://modelscope.cn/models/Qwen/Qwen3-4B-Instruct-2507) 下载；`scripts/download_model.py` 根据缓存的官方文件清单固定 snapshot revision，逐文件核对大小与 SHA-256。

GitHub 仓库包含官方文件清单 `configs/qwen3-4b-modelscope-files.json`，不包含模型权重。需要复现时运行 `python3 scripts/download_model.py`，即可按固定版本下载并校验模型。

采集器也可直接使用 Hugging Face ID 或本地模型目录。建议使用已下载的本地目录：

```bash
.venv/bin/python -m gqa_probe.capture --model models/Qwen3-4B-Instruct-2507 --output results/smoke_4k --context 4096 --samples 1 --decode-steps 2 --layers 0,17,35 --groups 0
.venv/bin/python -m gqa_probe.evaluate results/smoke_4k/traces --output results/smoke_4k/evaluation

.venv/bin/python -m gqa_probe.capture --model models/Qwen3-4B-Instruct-2507 --output results/pilot_16k --context 16384 --samples 2 --decode-steps 4 --layers 0,7,14,21,28,35
.venv/bin/python -m gqa_probe.evaluate results/pilot_16k/traces --output results/pilot_16k/evaluation
```

输出目录必须尚不存在，避免覆盖已有结果。GPU 生成遇 EOS 立即停止，不为了增加 Trace 数量继续生成 EOS 后的状态。

## KV4 数值参考的明确边界

这是一条 **Dense KV4 量化解量化参考路径**：所有层在 RoPE 后将新增 K 和 V 量化再解量化，缓存和 Attention 实际使用这些值。按每 Token、每 KV Head、每 32 通道分组；对称范围 [-7,7]，zero-point=0，FP16 scale、nearest-even rounding；没有未量化残留窗口。Scale 最小为 FP16 最小正 subnormal。

模型权重保持 BF16，缓存为了参考计算存储为解量化 BF16。Prefill 使用 SDPA；Decode 的 QK/Softmax/AV 使用 FP32，输出再转回 BF16。导出的 Logits 和 V 就是该计算路径的数值；离线评价用 FP64 重新计算稳定 Softmax，Tiny Qwen3 测试核对其与实际 Dense Attention 输出一致。

**尚未实现 INT4 打包缓存、实际 Scale 布局/访问计费或部署内核。** 本路径只支持选择质量研究，不能报告真实 KV4 存储量、LPDDR 流量减少或加速比。它不是 BF16 Trace 事后量化，也不是 KIVI/Quest 的复现。

## 文件与结果

- `gqa_probe/selection.py`：A1/A3/A3-B、保护页与逐批审计。
- `gqa_probe/attention.py`：稳定 Page Mass、Dense/Sparse 输出。
- `gqa_probe/capture.py`：量化参考缓存、自定义检索提示、GPU Trace。
- `gqa_probe/evaluate.py`：参数扫描、逐记录 JSONL、汇总 JSON/CSV。
- `tests/`：退化、不满页、预算、极端分数与真实小模型接口测试。
- `results/*/traces/manifest.json`：模型来源/哈希、代码哈希、量化规则、版本、样本及每条 Trace 信息。
- `results/*/traces/sample_*_input_ids.npy`：精确输入 Token IDs。
- `results/*/evaluation/records.jsonl`：包含每个 Head、每个批次的细节。
- `results/*/evaluation/summary.csv`：便于进一步统计和绘图。

GitHub 上传版本保留脚本、测试、研究文档、汇总 CSV/JSON、日志、审计结果及 `trace_manifest.json`。完整逐组/逐批结果压缩为 `results/*/evaluation/records.jsonl.gz`，解压内容与本地原始 JSONL 完全一致。模型权重、虚拟环境、下载缓存、原始 Q/K/V Trace 和未压缩的大体积 JSONL 均通过 `.gitignore` 排除；原始 Trace 继续保存在工作站。

可用 Python 标准库读取压缩明细：

```python
import gzip
import json

with gzip.open("results/pilot_16k/evaluation/records.jsonl.gz", "rt") as stream:
    for line in stream:
        record = json.loads(line)
        # 分析每个样本、层、步、KV Group 及参数组合。
```

Trace 格式：`logits[H,T]` 为已缩放且有限的全有效历史 QK 分数，`values[T,D]` 为同一 KV Head 的共享 V，`dense_output[H,D]` 为实际 Dense FP32 Attention 输出。当前采集仅支持 batch=1、连续有效历史、无 padding，不支持滑动窗口/任意非连续物理映射。

`selector_python_seconds` 只测 Python 参考选择器耗时；不包含模型、摘要或完整系统成本，不能当作硬件延迟。排序构建、动态轮次与 Page Gain 评估数也不能直接折算为 ASIC cycles。

当前工作量计数只完整提供动态轮次、Gain 评估数、实际选页和逐批信息；尚未提供排序/比较次数、候选列表访问次数、覆盖累加及缺口刷新次数。分批成本分析目前只能使用前述有限计数，不能声称已完成 V1.1 §4.2 的全部成本计费。Qwen2 架构路径尚未执行集成测试，次模型结果需单独验证后再报告。

## 结论限制和下一阶段

当前提示为固定随机种子的**自定义键值检索探索集**，不属于官方 RULER。短上下文冒烟和少量 16K Trace 用于发现实现问题、观察机制及生成后续假设，不能代替正式 RULER/LongBench 评价或直接放行 Gate 1。

后续正式实验需要固定独立样本、量化方案、任务与 Gate 判据，在更多样本、层、步与全部 KV Groups 上复核，按 Prompt 配对估计不确定性；再依文档推进 A3-C/A4/A4-B。BF16 参考、不同 KV4 方案和闭环 Sparse Decode 应分别保留标签。RTL/PPA 只在文档要求的条件成立后启动。
