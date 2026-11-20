"""Plot held-out INT6 quality against modeled per-token and answer reads."""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("report",type=Path)
    args=parser.parse_args()
    with (args.report/"quality.csv").open() as f:
        data=list(csv.DictReader(f))
    primary=next(r for r in data if r["policy"]=="mass_0.05_1000")
    fig,axes=plt.subplots(1,2,figsize=(9.4,4),layout="constrained",sharey=True)
    styles=(("all","Static full V","#555555","o"),("mass","Cumulative + refill","#146b91","s"),
            ("blasst","BLASST + refill","#ad492f","^"),("sparta","SPARTA + refill","#5d4b9a","D"))
    for ax,metric,units,scale in zip(axes,("read_per_decode","read_per_answer"),
                                    ("MB per decode token","GB per answer"),(1e6,1e9)):
        for mode,label,color,mark in styles:
            rows=sorted((r for r in data if r["mode"]==mode),key=lambda r:float(r[metric]))
            x=[float(r[metric])/scale for r in rows]
            y=[100*float(r["official_score"]) for r in rows]
            ax.plot(x,y,marker=mark,label=label,color=color,linewidth=1.3,markersize=5)
            for r,path,score in zip(rows,x,y):
                display=f'{int(r["budget_bp"])/100:g}%' if mode!="mass" else f'{100*float(r["threshold"]):g}% loss'
                ax.annotate(display,(path,score),xytext=(3,5),textcoords="offset points",fontsize=7,color=color)
        ax.set_xlabel(units)
        ax.grid(alpha=.22)
    x=float(primary["read_per_decode"])/1e6
    axes[0].axvspan(x/1.01,x/.99,color="#146b91",alpha=.09,zorder=0)
    axes[0].set_ylabel("Official retrieval score (%)")
    axes[0].set_ylim(95.65,99.14)
    axes[1].legend(loc="lower left",frameon=False,fontsize=7.8)
    fig.savefig(args.report/"quality_read_tradeoff.png",dpi=180)
    svg=args.report/"quality_read_tradeoff.svg"
    fig.savefig(svg)
    svg.write_text("\n".join(s.rstrip() for s in svg.read_text().splitlines())+"\n")


if __name__=="__main__":
    main()
