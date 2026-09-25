"""
plot_correlation.py - 32x32 estimated cross-event correlation matrix.

Design notes
------------
* The data's job is POLARITY (correlations are signed), so the colour scale is
  diverging: two hues with a neutral midpoint, never a rainbow.  RdBu_r is a
  ColorBrewer diverging ramp that is both colour-vision-deficiency safe and
  legible in greyscale print.
* The scale is forced symmetric about zero (vmin = -vmax) so the neutral colour
  lands exactly on rho = 0.  An asymmetric scale would put a hue at the
  midpoint and misrepresent the sign structure.
* The unit diagonal is masked.  Off-diagonal estimates span about +/-0.41; if
  the diagonal were drawn at 1.0 it would set the scale and flatten everything
  that matters.
* Events are ordered by category so block structure is visible without needing
  32 legible tick labels in a PRL-width column.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# PRL single-column width
COL_IN = 3.4


SHORT = {"Science and Technology": "Sci/Tech", "Entertainment": "Entmt",
         "Financials": "Fin", "Elections": "Elect", "Economics": "Econ",
         "Companies": "Cos", "Politics": "Pol", "Crypto": "Crypto",
         "Sports": "Sports"}


def _panel(ax, M, cats, norm, cmap, title, show_y):
    E = len(cats)
    im = ax.imshow(M, cmap=cmap, norm=norm, interpolation="nearest")

    labels, start = [], 0
    for i in range(1, E + 1):
        if i == E or cats[i] != cats[start]:
            labels.append((cats[start], (start + i - 1) / 2.0))
            if i < E:
                ax.axhline(i - 0.5, color="white", lw=0.7)
                ax.axvline(i - 0.5, color="white", lw=0.7)
            start = i

    ticks = [p for _, p in labels]
    names = [SHORT.get(c, c[:7]) for c, _ in labels]
    ax.set_xticks(ticks); ax.set_xticklabels(names, rotation=90, fontsize=4.5)
    if show_y:
        ax.set_yticks(ticks); ax.set_yticklabels(names, fontsize=4.5)
    else:
        ax.set_yticks([])
    ax.tick_params(length=0, pad=1.2)
    for s in ax.spines.values():
        s.set_linewidth(0.5); s.set_color("#999999")
    ax.set_title(title, fontsize=6, pad=3)
    return im


def build(ev_csv="event_correlation.csv", shr_csv="event_correlation_shrunk.csv",
          contracts_csv="contracts_used.csv", out="fig_correlation", T=89,
          delta=None):
    """
    Two panels, one shared colour scale.

    The scale is taken from the RAW estimates and reused for the shrunk panel.
    Rescaling each panel to its own range would make the shrunk matrix look just
    as noisy as the raw one -- the whole point is that its dispersion is smaller,
    so the comparison is only honest on a common scale.
    """
    raw = pd.read_csv(ev_csv, index_col=0)
    shr = pd.read_csv(shr_csv, index_col=0)
    con = pd.read_csv(contracts_csv)
    cat = con.groupby("event_ticker")["category"].first().to_dict()

    order = sorted(list(raw.index), key=lambda e: (cat.get(e, "zzz"), e))
    A = raw.loc[order, order].to_numpy(float).copy()
    B = shr.loc[order, order].to_numpy(float).copy()
    cats = [cat.get(e, "?") for e in order]
    E = len(order)

    np.fill_diagonal(A, np.nan)
    np.fill_diagonal(B, np.nan)

    vmax = float(np.nanmax(np.abs(A)))           # shared scale, from the raw panel
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#f2f2f2")

    fig, axes = plt.subplots(1, 2, figsize=(2 * COL_IN, COL_IN * 1.06))
    se = 1.0 / np.sqrt(max(T - 3, 1))

    _panel(axes[0], A, cats, norm, cmap,
           rf"(a) raw estimate  (sd $={np.nanstd(A):.3f}$)", True)
    dtxt = "" if delta is None else rf",  $\delta={delta:.2f}$"
    im = _panel(axes[1], B, cats, norm, cmap,
                rf"(b) shrunk  (sd $={np.nanstd(B):.3f}${dtxt})", False)

    cb = fig.colorbar(im, ax=axes, fraction=0.028, pad=0.015)
    cb.set_label(r"event-pair correlation  $\hat\rho$", fontsize=6)
    cb.ax.tick_params(labelsize=5, length=2)
    cb.outline.set_linewidth(0.4)

    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    finite = A[np.isfinite(A)]
    return dict(events=E, pairs=E * (E - 1) // 2, vmax=round(vmax, 4), se=round(se, 4),
                raw_sd=round(float(np.nanstd(A)), 4),
                shrunk_sd=round(float(np.nanstd(B)), 4),
                above_2se=int((np.abs(finite) > 2 * se).sum()) // 2,
                categories=len(set(cats)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="event_correlation.csv")
    ap.add_argument("--shrunk", default="event_correlation_shrunk.csv")
    ap.add_argument("--delta", type=float, default=None)
    ap.add_argument("--contracts", default="contracts_used.csv")
    ap.add_argument("--T", type=int, default=89)
    ap.add_argument("--out", default="fig_correlation")
    a = ap.parse_args()

    info = build(a.events, a.shrunk, a.contracts, out=a.out, T=a.T,
                 delta=a.delta)
    print(f"  wrote {a.out}.pdf and {a.out}.png")
    for k, v in info.items():
        print(f"    {k:12s} {v}")


if __name__ == "__main__":
    main()
