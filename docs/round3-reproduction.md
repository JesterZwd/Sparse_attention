# 第三轮实验复现

实验基线为 `15c100bda2ab78f9a1aa9ff56ec34a3a60f519ee`。实验使用 Qwen3-4B-Instruct-2507 和 RTX 3090。模型沿用第二轮的本地校验版本，RULER 固定在 `db43f92ff634bbce1c33d1fb15858aec4bba509a`。32K 指输入和生成上限的总和。VT、Multi-Value 和备用 FWE 分别留出 30、128 和 50 个生成 token。

## 布局和字节预算

每个 KV 组的每个 Token 有 128 B 的 K/V 码、16 B 的 FP16 Scale 和 8 B 的 uint8 Zero Point。每页分别将 K、V 的码、Scale、Zero Point 段上舍入到 16 B，最后一页按有效长度计算。Dense 读基数采用同一布局。摘要向量使用 INT8 及每向量 FP16 Scale，页计数占 1 B，摘要记录按 16 B 对齐。单质心每页记录 144 B，Quest 极值对为 272 B。每个选中页计 4 B 局部索引，按连续索引条目的逻辑读取计费。该布局对每层每 KV 组分别计费。

每步扫描全部摘要，包括开头 32 和最近 128 个 Token 对应的保护页。去重并记录物理页扩张。最终页总被保护，其余可选页的布局成本相同。先扣摘要、保护页和这些页的索引，再分配整页额度。共同上限为同一步 packed Dense 读取量的 10%、15% 或 20%。另算统一 4 MiB SRAM 的热态静态驻留，按模型的层与 KV 组均分，每份从 Page ID 最小的封存页开始驻留，活动尾页按外存读写。账本单列预填充写入及每 Decode 步摘要更新写入。驻留列是布局推算，忽略外存突发与冷热替换开销。

## 工作站命令

```bash
.venv/bin/python -m scripts.prepare_ruler --config configs/ruler_round3_16k.json --output results/round3_reproduction/data/16k
.venv/bin/python -m scripts.prepare_ruler --config configs/ruler_round3_32k.json --output results/round3_reproduction/data/32k
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u -m scripts.run_ruler_round3 --root results/round3_reproduction
```

使用项目的 RULER 依赖和上一轮准备的固定语料。备用 FWE 额外依赖 `scipy==1.16.3`。运行器在 Dense BF16 的 VT 校准分数低于 90% 时启动 FWE。源代码、模型、配置和数据校验值相同时可用 `--resume` 恢复到首个缺失提示。预填充只做两次，先分别保存 BF16 和 affine32 前缀。同一提示下的政策复用 affine32 前缀及摘要，Decode 更新只接收当前 Token 的 RoPE 后解量化 Key。

输出目录包含精确数据 Manifest、每个 Cohort 的配置和完整清单，以及校准报告。逐提示同轨迹诊断在 Dense KV4 的抽样层与步上计算绝对及相对误差分位数、真实保留质量、与同页数 Oracle 的重合及可核验的金标准页召回。VT 和 FWE 目前提供官方任务分数和完整命中率。首个答案 Token 来自 Dense Prefill，并计入生成上限。

质量结果来自全部层使用的 Mask 参考。在摘要政策中，Query 和摘要先决定选页，随后参考后端计算全宽 QK 和带掩码的 AV。抽样 Gather 另行验证数值一致。报告的字节量按上面的布局计算，Prefill/Decode 计时属于 PyTorch 参考路径。正式外存、打包内核和 SRAM 访问成本需要读取原型验证。
