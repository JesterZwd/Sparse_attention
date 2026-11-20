# Round 6 子页读取与头窗口复现

基线 `d5f0b612291fd7fa2836afa9d7640caa3440adaf`。复用已校验的 Qwen3-4B-Instruct-2507 权重、固定 RULER v1 生成器和语料。运行环境为 RTX 3090、Python 项目环境和 C++17 编译器。正式流程用一个 affine32 Prefill，在同一提示上分叉生成。公开清单只保存输入校验和统计。

```bash
.venv/bin/python -m scripts.prepare_ruler --config results/round6_subpage_head/prepare_calibration.json --output results/round6_reproduction/data/calibration
.venv/bin/python -m scripts.prepare_ruler --config results/round6_subpage_head/prepare_test.json --output results/round6_reproduction/data/test
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u -m scripts.round6_pipeline --root results/round6_reproduction
```

正式结果目录用 `results/round6_subpage_head`。定点窗口补充回放命令为 `.venv/bin/python -m scripts.round6_window_targeted`，最终回执和三账总表由 `.venv/bin/python -m scripts.audit_round6_results` 导出。回放保留拟合得到的头掩码，采样位置补到三个入选的层和组；不重新拟合。

地址回放依赖 `sim/round6_lpddr5.cpp`，Python 首次编译到本机缓存。采用 32 B BL16、x16 单通道、16 banks、2 KiB row、四个 bank group、open row、FIFO 和 32 个未完成 burst 的读模型，按 V 发出前的摘要/K/门控屏障串接。时序参数来自固定的 [Ramulator 2.1 LPDDR5-6400 preset](https://github.com/CMU-SAFARI/ramulator2/blob/7bead6cb7c4b03e095c6d4769ae3181a8e0bf6da/python/ramulator/dram/lpddr5.py)。模型按组独立清空行状态，包含读延迟、行切换和 bank group 间距，省略刷新、并发组交织、读写反转和实板测量。算子吞吐来自固定架构假设，共享同一个 CK 标尺。

`protocol.json` 在校准前冻结数据和路径，`finished.json` 记录为何停在校准。无过筛候选时不启动 128 条主测试。组周期原始行、分项字节、预算超限位置和评分在 `report/` 下。设计推断与每条闭环预测分别保留。
