# plot_scaling.py - time-to-solution vs universe size, for the paper.

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "scaling_results.csv"
OUT = "fig_scaling.pdf"

STYLE = {
    "SA":              dict(color="#1f77b4", marker="o", ls="-",  label="Simulated annealing"),
    "BruteForce2n":    dict(color="#7f7f7f", marker="s", ls=":",  label=r"Brute force ($2^n$)"),
    "EnumCardinality": dict(color="#2ca02c", marker="^", ls="--", label=r"Exhaustive $\binom{n}{K}$"),
    "CBC-linearized":  dict(color="#d62728", marker="v", ls="-.", label="CBC (linearized QUBO)"),
    "CBC-native":      dict(color="#ff7f0e", marker="D", ls="-",  label="CBC (native MIQP)"),
}


def main():
    df = pd.read_csv(CSV)
    df = df[df["status"] == "ok"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # ---- left: wall-clock vs n -------------------------------------------
    for solver, st in STYLE.items():
        sub = df[df["solver"] == solver].sort_values("n")
        if sub.empty:
            continue
        ax.plot(sub["n"], sub["time_s"], ms=4, lw=1.4, **st)
        # mark runs that hit the MILP time limit (not proved optimal)
        to = sub[(~sub["proved_optimal"]) & (sub["solver"].str.startswith("CBC"))]
        if not to.empty:
            ax.scatter(to["n"], to["time_s"], s=90, facecolors="none",
                       edgecolors=st["color"], lw=1.4, zorder=5)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"universe size $n$")
    ax.set_ylabel("time to solution (s)")
    ax.grid(alpha=0.3, which="both", lw=0.5)
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    ax.set_title("(a) solver cost", fontsize=9)

    # ---- right: optimality gap of SA vs best known ------------------------
    best = df.groupby("n")["energy"].min().rename("best")
    m = df.merge(best, on="n")
    m["gap"] = np.where(m["best"].abs() > 0,
                        (m["energy"] - m["best"]).abs() / m["best"].abs() * 100, 0.0)
    for solver in ["SA", "CBC-linearized", "CBC-native"]:
        sub = m[m["solver"] == solver].sort_values("n")
        if sub.empty:
            continue
        st = dict(STYLE[solver]); st.pop("ls")
        ax2.plot(sub["n"], sub["gap"], ms=4, lw=1.4, ls="-", **st)

    ax2.set_xscale("log")
    ax2.set_xlabel(r"universe size $n$")
    ax2.set_ylabel("gap vs best known (\\%)")
    ax2.grid(alpha=0.3, which="both", lw=0.5)
    ax2.legend(fontsize=6, frameon=False, loc="upper left")
    ax2.set_title("(b) solution quality", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print("wrote", OUT)

    # ---- LaTeX table -----------------------------------------------------
    piv_t = df.pivot_table(index="n", columns="solver", values="time_s")
    piv_e = df.pivot_table(index="n", columns="solver", values="energy")
    print("\n--- time (s) ---\n", piv_t.round(3).to_string())
    print("\n--- energy ---\n", piv_e.round(4).to_string())

    order = ["EnumCardinality", "CBC-native", "CBC-linearized", "SA"]
    order = [c for c in order if c in piv_t.columns]
    lines = []
    for n, row in piv_t.iterrows():
        K = int(df[df["n"] == n]["K"].iloc[0])
        pairs = int(df[df["n"] == n]["excl_pairs"].iloc[0])
        cells = []
        for c in order:
            v = row.get(c, np.nan)
            cells.append("---" if pd.isna(v) else f"{v:.2f}")
        lines.append(f"{n} & {K} & {pairs} & " + " & ".join(cells) + r" \\")
    print("\n--- LaTeX rows (n, K, pairs, " + ", ".join(order) + ") ---")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
