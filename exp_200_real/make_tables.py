"""
make_tables.py - emit LaTeX for the paper tables that cannot be printed row-by-row.

tab:contracts and tab:correlations were written for a 12-contract universe with
4 events, where every contract and every correlation fit on the page.  At 167
contracts / 32 events / 496 estimated event-pair correlations they have to
become summaries.  Both are regenerated from the CSVs so they stay in sync with
whatever the pipeline last produced.
"""

import argparse
from math import comb

import numpy as np
import pandas as pd

COV_FORM = {
    "disjoint":     r"$-p_i p_j$",
    "nested":       r"$\min(p_i,p_j)-p_i p_j$",
    "unstructured": r"estimated",
}


def _esc(s: str) -> str:
    return str(s).replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


def table_contracts(contracts_csv="contracts_used.csv"):
    df = pd.read_csv(contracts_csv)
    n, E = len(df), df["event_ticker"].nunique()

    L = []
    L.append(r"\begin{table}[b]")
    L.append(rf"\caption{{\label{{tab:contracts}}Composition of the {n}-contract "
             rf"Kalshi universe ({E} events).  Contracts are drawn whole-event-at-a-"
             r"time so that event structure is preserved; the structure class fixes "
             r"the within-event covariance analytically, leaving only cross-event "
             r"terms to be estimated.}")
    L.append(r"\begin{ruledtabular}")
    L.append(r"\begin{tabular}{lccc}")
    L.append(r"Structure & Events & Contracts & Within-event covariance \\")
    L.append(r"\colrule")
    for s in ["disjoint", "nested", "unstructured"]:
        g = df[df["structure"] == s]
        if not len(g):
            continue
        L.append(f"{s.capitalize()} & {g['event_ticker'].nunique()} & {len(g)} & "
                 f"{COV_FORM[s]} " + r"\\")
    L.append(r"\colrule")
    L.append(rf"Total & {E} & {n} & \\")
    L.append(r"\end{tabular}")
    L.append(r"\end{ruledtabular}")
    L.append(r"\end{table}")
    latex_a = "\n".join(L)

    # category breakdown as a second, narrower table
    C = []
    C.append(r"\begin{table}[b]")
    C.append(rf"\caption{{\label{{tab:categories}}Category composition of the "
             rf"{n}-contract universe.  Prices are live Kalshi mid-quotes.}}")
    C.append(r"\begin{ruledtabular}")
    C.append(r"\begin{tabular}{lccc}")
    C.append(r"Category & Events & Contracts & Price range \\")
    C.append(r"\colrule")
    for cat, g in sorted(df.groupby("category"), key=lambda kv: -len(kv[1])):
        C.append(f"{_esc(cat)} & {g['event_ticker'].nunique()} & {len(g)} & "
                 f"{g['price'].min():.2f}--{g['price'].max():.2f} " + r"\\")
    C.append(r"\end{tabular}")
    C.append(r"\end{ruledtabular}")
    C.append(r"\end{table}")
    return latex_a, "\n".join(C)


def table_correlations(ev_csv="event_correlation.csv",
                       contracts_csv="contracts_used.csv", T=89, top=6):
    ev = pd.read_csv(ev_csv, index_col=0)
    df = pd.read_csv(contracts_csv)
    cat = df.groupby("event_ticker")["category"].first().to_dict()

    E = len(ev)
    M = ev.to_numpy(float)
    iu = np.triu_indices(E, 1)
    off = M[iu]
    se = 1.0 / np.sqrt(max(T - 3, 1))

    names = list(ev.index)
    ranked = sorted(((M[i, j], names[i], names[j]) for i, j in zip(*iu)),
                    key=lambda t: -abs(t[0]))[:top]

    L = []
    L.append(r"\begin{table}[b]")
    L.append(rf"\caption{{\label{{tab:correlations}}Cross-event correlations "
             rf"estimated from {T} daily price increments, replacing a hand-set "
             rf"correlation table.  Each event contributes one factor (first "
             rf"principal component of its contracts' increments), giving "
             rf"$\binom{{{E}}}{{2}}={comb(E,2)}$ estimated correlations in place of "
             rf"$\binom{{{len(df)}}}{{2}}={comb(len(df),2):,}$ free contract pairs.  "
             rf"The sampling standard error is $1/\sqrt{{T-3}}={se:.3f}$, so "
             rf"individual estimates are weak; the strongest are shown.}}")
    L.append(r"\begin{ruledtabular}")
    L.append(r"\begin{tabular}{lcr}")
    L.append(r"\multicolumn{3}{c}{\textit{Distribution of the "
             + str(len(off)) + r" estimates}} \\")
    L.append(r"\colrule")
    L.append(rf"mean & & ${off.mean():+.3f}$ \\")
    L.append(rf"standard deviation & & ${off.std():.3f}$ \\")
    L.append(rf"range & & ${off.min():+.3f}$ to ${off.max():+.3f}$ \\")
    L.append(rf"$|\rho|>2\,\mathrm{{s.e.}}$ & & {int((np.abs(off)>2*se).sum())} "
             rf"of {len(off)} \\")
    L.append(r"\colrule")
    L.append(r"\multicolumn{3}{c}{\textit{Strongest estimated pairs}} \\")
    L.append(r"\colrule")
    L.append(r"Event pair & Categories & $\hat\rho$ \\")
    L.append(r"\colrule")
    for r, a, b in ranked:
        ca, cb = cat.get(a, "?")[:11], cat.get(b, "?")[:11]
        L.append(f"{_esc(a[:22])} -- {_esc(b[:22])} & {_esc(ca)}/{_esc(cb)} & "
                 f"${r:+.3f}$ " + r"\\")
    L.append(r"\end{tabular}")
    L.append(r"\end{ruledtabular}")
    L.append(r"\end{table}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contracts", default="contracts_used.csv")
    ap.add_argument("--events", default="event_correlation.csv")
    ap.add_argument("--T", type=int, default=89)
    ap.add_argument("--out", default="tables_generated.tex")
    a = ap.parse_args()

    struct, cats = table_contracts(a.contracts)
    corr = table_correlations(a.events, a.contracts, T=a.T)

    blob = "\n\n".join([
        "% ---- replaces tab:contracts (12-row contract listing) ----",
        struct,
        "% ---- new: category breakdown ----",
        cats,
        "% ---- replaces tab:correlations (hardcoded rho table) ----",
        corr,
    ])
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(blob + "\n")

    print(blob)
    print(f"\n% wrote {a.out}")


if __name__ == "__main__":
    main()
