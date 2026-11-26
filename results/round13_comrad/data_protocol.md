# CoMRAD Round 13 数据口径

Qwen3 64K Multi-Value 复用 Round 12 封存的 64 道独立提示。封存收据核对原数据、原实验结论和全部实例哈希。历史比对使用工作站上已完成的官方版本 RULER 数据清单，覆盖实际使用过的批次及其已物化的相邻分组。按实例哈希而非跨数据集重复的序号识别提示。 [64K 封存收据](qwen3_64k_seal.json)

32K FWE 采用 8 条校准和 64 条独立提示。Qwen2.5-7B Multi-Value 单列 4 条掩码拟合、16 条校准和 64 条独立提示。各分组使用不同种子，先应用实际聊天模板并预留生成长度，再核对 32K 上限。Qwen2.5 使用 32768 原生位置上限及上游同版本 tokenizer。CWE 的配置预留 120 个生成 token，按 FWE 校准结果决定是否生成。生成器代码为 Apache‑2.0，Paul Graham 文章保留其原始版权。 [RULER 官方版本](https://github.com/NVIDIA/RULER/tree/db43f92ff634bbce1c33d1fb15858aec4bba509a) · [Qwen2.5 模型版本](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/tree/a09a35458c702b33eeacc393d103063234e8bc28)

LongBench v2 用官方无思维链的直接选择题提示，套用 Qwen3 聊天模板并保留原文。在 503 道官方题中，按实际输入 token 数筛选至少 16384，输入与 128 个生成预留之和至多 32768。固定种子和原始题 ID 决定选题顺序，具体 ID、长度和哈希见收据。论文对照以贪心解码控制各方案，使用官方答案提取逻辑与字母精确匹配，无法提取答案计为错误。结果记录为这个固定子集的准确率。上游数据卡声明 Apache‑2.0，评测代码采用 MIT。发布包保留版本、哈希和结果，原文及适配输入保存在工作站。 [LongBench v2 数据卡](https://huggingface.co/datasets/zai-org/LongBench-v2/blob/2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9/README.md) · [冻结版本的提示模板](https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/prompts/0shot.txt) · [冻结版本的评分脚本](https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/pred.py)

## 复现命令

```bash
.venv/bin/python -m scripts.prepare_round13_data workspace
```

准备好模型及 RULER 来源后，脚本生成缺失的数据集并核对发布时的冻结哈希。已从前一台工作站复制输入时可用 `verify-workspace` 校验。LongBench 原始文件从固定版本的上游地址下载。 [环境与迁移步骤](../../docs/round13-workstation.md)
