# Minimal Probe Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. User requested direct execution on the workstation with one RTX 3090.

**Goal:** 实现并运行 V1.1 §5.1 的 A1/A3/A3-B 最小探针，保留可复现结果，区分正确性检查、真实 Trace 与任务质量结论。

**Architecture:** NumPy 离线选择器与统计模块；PyTorch/Transformers 在 GPU 上运行同一条 Dense KV4 数值参考路径并导出分组 Trace；命令行扫描相同 Trace，写出逐组明细和汇总。先短上下文冒烟，再 16K；未通过 Gate 1 不启动候选前沿和 RTL。

**Tech Stack:** 系统 Python 3.12、项目内 .venv、PyTorch CUDA 12.1、Transformers（固定版本支持 Qwen3）、NumPy、pytest。

**Spec:** ../../../端侧LLM_GQA联合TopKTopP_研究与实验方案_V1.1.md （从项目根目录读取该文件）

## Global Constraints

- 比较 A1、A3、A3-B(b=1/4/8/16)；同一物理预算、同一保护、同一 V。
- Page Size=32；Sink=1 Page；Local=128 Tokens；预算 5%/10%/20%；目标 0.7/0.8/0.9。
- 固定预算达标后按预建 A1 列表补齐；补页排除动态改变率统计。
- 逐 Head 全历史归一化；Sparse 输出从选中原始 Logits 重算稳定 Softmax。
- BF16 参考、KV4 数值仿真、压缩 KV4 存储和闭环稀疏模型分别标记，不互称。
- 保护页去重计入预算；无效预算报错；不满尾页与 Local 跨页必须正确。
- 原始论文方案不修改；不更新系统驱动；依赖和下载放项目目录。

## Review Focus

1. Local=128 在长度非页对齐时跨越五页，不能总按四页处理。
2. 相同收益按较小 Page ID 打破平局，p=1 与 b=1 退化成立。
3. 非空选集在 Dense 概率下溢时仍应输出有限值。
4. 必须在模型内部使用量化解量化后的 K/V；仅事后量化 Trace 不能冒充 Dense KV4 路径。
5. 小样本、重复文本和自定义检索提示仅作冒烟/探索；不得据此放行 Gate 1 或声称 RULER 分数。

### Task 1: 环境与选择核心

**Files:** `requirements.txt`, `.gitignore`, `gqa_probe/selection.py`, `tests/test_selection.py`。

**Interfaces:** `select_pages(mass[J,H], protected, budget, target, algorithm, batch_size=1, mode='fixed') -> SelectionResult`；返回选页顺序、覆盖及逐批审计。

- [x] 创建 .venv，安装固定版本，运行 CUDA 张量运算记录 GPU/版本。
- [x] 先写测试：p=1 排序退化、b=1 退化、保护去重、预算错误、早停补页、确定性平局、批内冗余和有效批率。
- [x] 运行测试确认功能缺失，最小实现，再运行全部测试。

### Task 2: 稳定 Attention 与离线报告

**Files:** `gqa_probe/attention.py`, `gqa_probe/evaluate.py`, `tests/test_attention.py`。

**Interfaces:** `page_mass(logits[H,T], page_size) -> [J,H]`；`attention_output(logits[H,T], values[T,D], pages, page_size) -> [H,D]`；评估输出 CSV/JSON。

- [x] 先测试尾页、极端 Logits、非法输入、全页输出一致、相同 Value 数据路径。
- [x] 实现稳定 Softmax；汇总平均/尾部覆盖、缺口、绝对/相对输出误差、批内冗余和工作量。
- [x] 合成用例只验证正确性，输出明确标注 synthetic。（合成数据仅存在于测试，没有冒充实测的结果目录。）

### Task 3: GPU Dense KV4 Trace 与真实探针

**Files:** `gqa_probe/capture.py`, `tests/test_capture.py`, `README.md`, `results/`。

**Interfaces:** 逐记录保存 logits、同路径 V、样本/层/步/组/量化元数据；离线评估只消费这些记录。

- [x] 验证 GPU、模型下载与官方配置，固定模型 revision。
- [x] 先用随机小 Qwen3 模型验证量化进入实际 Attention 和 Trace 的一致性。
- [x] 量化无现成实现时，使用逐 Token、逐 Head、32 通道一组的对称 INT4 数值仿真，FP16 scale；K 在 RoPE 后量化。明确未进行压缩存储或性能优化。
- [x] 跑短上下文真实模型冒烟，再跑 16K 探针；保留命令、日志、依赖版本、数据来源和结果。
- [x] 运行全部测试，复核结果标签，记录未完成的正式任务评估与 Gate 条件。

## Execution notes

- 当前目录不是可用 Git 仓库，`.git` 为只读占位目录；直接在用户指定目录实现，不改动 `.git`，不创建 worktree 或伪造提交。
- 用户已明确要求在工作站运行；沿用 V1.1 设计直接执行，不增加重复确认步骤。
