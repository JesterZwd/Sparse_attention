# RULER 16K 第二轮复现

实验比较固定 GQA 共享物理页预算下的 A1 与有限换页。主配置为 (p=0.8)、20% 页预算，压力配置为 (p=0.9)、10% 页预算。预先固定的检索、输出误差和选择耗时判据见执行计划与 `results/ruler_round2_protocol.json`。

工作站使用 Qwen3-4B-Instruct-2507、RTX 3090、PyTorch 2.5.1+cu121 和 Transformers 4.51.3。模型及环境安装说明见项目 README。数据采用 NVIDIA RULER v1 旧版流水线的固定提交 `db43f92ff634bbce1c33d1fb15858aec4bba509a`。`results/ruler_round2_data/manifest.json` 保存生成器、218 个语料来源、适配补丁、依赖和各数据划分校验值。工作站上的数据与生成器准备脚本保存在该目录的 `provenance/`，数据配置为 `configs/ruler_round2.json`。总长 16K 包含 Qwen 模板、官方答案前缀和最多 128 个生成 token。

先在旧 Trace 上重放，并在工程集比较量化路径。

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m gqa_probe.replay_round2 --output results/round2_replay
.venv/bin/python -m scripts.run_ruler_engineering
.venv/bin/python -m scripts.run_quantization_engineering
```

工程量化比较在相同的 32 个提示上选择 `affine32`。它用每 32 通道一组的 4 位仿射量化，在 Attention 路径使用解量化 BF16 KV。按 4 位 codes、FP16 scale 和 uint8 zero-point 计算，布局理论值为 4.75 bits/element。闭环实验比较量化数值与页选择对质量的作用。

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 .venv/bin/python -u -m gqa_probe.ruler_experiment --phase calibration --output results/ruler_round2_calibration --quantizer affine32
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 .venv/bin/python -u -m scripts.run_ruler_round2_test --wait
```

校准从 128 个提示的 Dense KV4 轨迹冻结一次候选。测试启动器核对校准产物哈希、重新计算候选，并核对预先登记的分析规则，随后用 512 个独立提示各跑六条完整生成路径。每个提示用两个 Dense Prefill，并在各 KV4 生成路径间共享已验证的前缀。最大输出 token 数包含 Prefill 产生的首个 token。Decode 在全部层和 KV Groups 使用相应选择策略。

每个正式实验目录保存 `config.json`、`progress.json`、源代码快照与逐提示检查点。报告脚本读取完成标记，校验预测和记录，输出含任务分数、完整正确率、金标准页或片段召回、配对误差及逐层尾部的表格。选择器耗时按固定阶层样本，从已归一化的 Page Mass 计到返回决定，另计比较、候选构建和覆盖更新。质量推理使用完整历史 QK 和带掩码的全宽 AV，运行时间表征这条参考路径。

重启中断的校准时，请在同一命令追加 `--resume`。正式测试中断后，从同一个校准目录执行测试启动器并追加 `--resume-test`。启动器核对源代码和配置后重用已保存的完整提示。如果报告独立中断，可指定新目录重新生成。

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m scripts.report_ruler_round2 --output results/ruler_round2_report_retry
```
