"""Plot the measured fixed-K quality and modeled V traffic trade-off."""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.1), layout="constrained", sharey=True)
    styles = [
        ("mass", "Cumulative page mass", "#146c93", "o"),
        ("blasst", "Running peak gate", "#ab422b", "s"),
        ("sparta", "Token Top-p union", "#6d519d", "^"),
        ("micro_4_addr", "Microbatch 4, address order", "#6d7417", "D"),
    ]
    for ax, (cohort, title) in zip(axes, (("diagnostic", "16K calibration"), ("32k_mv", "32K calibration"))):
        with (args.report / cohort / "fixed_k_curve.csv").open() as stream:
            observations = list(csv.DictReader(stream))
        for mode, label, color, marker in styles:
            rows = [r for r in observations if r["mode"] == mode]
            xs = [float(r["v_saved"])*100 for r in rows]
            ys = [float(r["absolute_p99"]) for r in rows]
            ax.plot(xs, ys, marker=marker, linewidth=1.45, markersize=4.5, label=label, color=color)
            if mode == "mass":
                for x, y, r in zip(xs, ys, rows):
                    ax.annotate(f'{100*float(r["threshold"]):g}%', (x,y), xytext=(-3,-11),
                                textcoords="offset points", fontsize=8, color=color)
        ax.scatter([0], [0], zorder=3, color="#343434", s=15)
        ax.grid(alpha=.22)
        ax.set(xlabel="Shared V read bytes saved (%)", title=title, xlim=(-2, 90), ylim=(-.05, 3.2))
    axes[0].set_ylabel("Prompt mean absolute error P99")
    axes[1].legend(loc="upper left", fontsize=7.8, frameon=False)
    fig.savefig(args.report / "fixed_k_quality_traffic.svg")
    fig.savefig(args.report / "fixed_k_quality_traffic.png", dpi=180)


if __name__ == "__main__":
    main()
