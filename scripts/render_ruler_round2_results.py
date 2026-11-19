"""Render the verified Round 2 report as Markdown after all 512 prompts finish."""

import argparse
import csv
import json
from pathlib import Path


def render(report, output):
    def load(name):
        return json.loads((report / name).read_text())

    decision = load("decision.json")
    audit = load("audit.json")
    intervals = load("paired_intervals.json")
    timing = load("timing_summary.json")
    with (report / "quality.csv").open() as stream:
        quality = [row for row in csv.DictReader(stream) if row["task"] == "all"]
    if audit["status"] != "passed" or audit["test_prompts"] != 512:
        raise ValueError("rendering requires a verified 512-prompt run")
    outcome = "有限换页达到本轮预设的质量与成本条件。" if decision["passed"] else (
        "有限换页未达到本轮预设的质量与成本条件，建议保留 A1 作为底座。")
    lines = [
        "# RULER 16K 独立测试结果", "", outcome, "",
        "独立测试包含 512 个提示，每子任务 128 个。候选为 δ=0.01、最多一次换页。"
        "全部方法采用 Dense Prefill，首个生成 token 来自 Prefill。稀疏方法的后续 Decode"
        "在 36 层和全部 8 个 KV Groups 使用相应选择器。选择依据完整历史 QK。", "",
        "| 配置 | 方法 | 官方分数 | 完整正确 | 闭环金标准页召回 | 闭环全目标片段保留 | 绝对 P95 | 绝对 P99 | 相对 P95 | 相对 P99 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def number(text, percent=False):
        if not text:
            return "—"
        value = float(text)
        return f"{value:.3%}" if percent else f"{value:.6f}"

    for row in quality:
        tails = [number(row["dense_kv4_replay_" + metric]) for metric in
                 ("absolute_error_p95", "absolute_error_p99", "relative_error_p95", "relative_error_p99")]
        fields = [
            row["setting"], row["method"], number(row["official_score"], True),
            f'{row["complete_correct"]}/{row["prompts"]}', number(row["closed_loop_gold_page_recall"], True),
            number(row["closed_loop_all_gold_spans_retained_rate"], True), *tails,
        ]
        lines.append("| " + " | ".join(fields) + " |")
    lines.extend(["", "尾部误差按 Dense KV4 轨迹计算，先求每个提示的分位数，再跨提示平均。"
                  "闭环召回使用各方法实际生成轨迹。", "",
                  "| 配对比较 | 指标 | 差值 | 97.5% 配对区间 |",
                  "|---|---|---:|---:|"])
    for setting, stats in intervals.items():
        for name in ("official_score", "complete_correct", "absolute_error_p99", "relative_error_p99"):
            if name not in stats:
                continue
            row = stats[name]
            if name.endswith("_p99"):
                delta, low, high = (row[k] for k in ("relative_delta", "relative_ci_low", "relative_ci_high"))
                delta_text = "—" if delta is None else f"{delta:.3%}"
                interval = "—" if low is None else f"[{low:.3%}, {high:.3%}]"
            else:
                delta_text = f'{100 * row["point_delta"]:+.3f} 个百分点'
                interval = f'[{100 * row["ci_low"]:+.3f}, {100 * row["ci_high"]:+.3f}] 个百分点'
            lines.append(f"| {setting} | {name} | {delta_text} | {interval} |")
    lines.extend(["", "配对区间以 Prompt 为单位，在子任务内重采样。两条预设质量判据"
                  "分别结合完整正确率和绝对误差 P99。点估计与完整的配对胜负计数见 `paired_intervals.json`。",
                  "", "| 配置 | 方法 | CPU 中位耗时（µs） | CPU P95（µs） | 中位数相对 A1 |",
                  "|---|---|---:|---:|---:|"])
    for row in timing["summary"]:
        config = f'p={row["target"]:g}, {row["budget_fraction"]:.0%}'
        lines.append(f'| {config} | {row["spec"]} | {1e6 * row["median_seconds"]:.2f} | '
                     f'{1e6 * row["p95_seconds"]:.2f} | {row["median_ratio_to_a1"]:.3f}× |')
    lines.extend(["", "CPU 耗时覆盖候选构建、选择和覆盖更新，输入是已归一化的 Page Mass。"
                  "计时方法保留轻量计数器，排序比较另外实测。完整条件和操作账本见 `decision.json` "
                  "及 `cost_records.jsonl.gz`。Attention 的 QK 与 AV 仍按完整历史执行，"
                  "因此此次测量对应数值参考实现的选择器成本。", ""])
    with output.open("x") as stream:
        stream.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("results/ruler_round2_report"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    render(args.report, args.output or args.report / "report.md")


if __name__ == "__main__":
    main()
