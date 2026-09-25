# select_contracts.py - pick a reviewable working set of contracts from the
# cached Kalshi universe and write it to CSV.
#
# Selection is event-at-a-time, never contract-at-a-time, because the structure
# that carries covariance information lives at the event level:
#   * mutually-exclusive event -> contracts are disjoint, Cov = -p_i p_j exactly
#   * threshold ladder         -> contracts are nested, Cov = min(p_i,p_j) - p_i p_j
# Splitting an event would throw that away.
#
# Events are ranked by traded volume: estimating a correlation empirically needs
# price history, and untraded markets have none.

import argparse
import numpy as np
import pandas as pd

from fetch_universe import load_universe

COHERENCE_TOL = 0.10    # |sum of YES prices - 1| allowed inside an MX event
PRICE_FLOOR = 0.02      # drop degenerate near-certain / near-impossible quotes
PRICE_CEIL = 0.98
MIN_EVENT_SIZE = 2
MAX_EVENT_SIZE = 15

# Events whose logical structure the exchange flag and the strike parser cannot
# see.  Checked against each market's settlement rules (rules_primary) on
# 2026-09-24; titles alone are ambiguous (both Greenland markets are titled
# "Will Trump buy Greenland?", and KXALIENS-27-29 is titled "in 2029" but
# settles on "before Jan 20, 2029").
#   disjoint: exactly one outcome can settle YES, but Kalshi does not set
#             mutually_exclusive, so without this the penalty never applies.
#   nested:   deadline ladders ("by 2027" implies "by 2029"); the strike is a
#             date, not a number, so _classify cannot order them.
STRUCTURE_OVERRIDES = {
    "KXVENEZUELALEADER-26DEC31": "disjoint",  # one head of state on Dec 31, 2026
    "KXGREENLAND-29": "nested",               # before Jan 1 2027 => before Jan 20 2029
    "KXTRUMPOUT27-27": "nested",              # out before 2027 implies before 2028
    "KXALIENS-27": "nested",                  # confirmed before Nov 2026 => ... => before Jan 20 2029
}


def _strike(ticker: str) -> float:
    """trailing numeric strike, if the ticker encodes one"""
    for marker in ("-T", "-B"):
        if marker in ticker:
            tail = ticker.rsplit(marker, 1)[-1]
            try:
                return float(tail)
            except ValueError:
                pass
    # price thresholds with no marker, e.g. KXBTCMAXY-26DEC31-119999.99.  The
    # decimal point is required so that date suffixes like -27 are not read
    # as strikes.
    tail = ticker.rsplit("-", 1)[-1]
    if "." in tail:
        try:
            return float(tail)
        except ValueError:
            pass
    return float("nan")


def _classify(grp: pd.DataFrame) -> str:
    """label an event by the covariance structure it implies"""
    override = STRUCTURE_OVERRIDES.get(grp["event_ticker"].iloc[0])
    if override:
        return override
    if bool(grp["mutually_exclusive"].iloc[0]):
        return "disjoint"
    strikes = grp["strike"].to_numpy(float)
    if np.isfinite(strikes).all() and len(set(strikes)) == len(strikes):
        # a threshold ladder prices monotonically against the strike
        order = np.argsort(strikes)
        p = grp["price"].to_numpy(float)[order]
        if np.all(np.diff(p) <= 1e-9) or np.all(np.diff(p) >= -1e-9):
            return "nested"
    return "unstructured"


def select(univ: pd.DataFrame, n_target: int = 200, seed: int = 42) -> pd.DataFrame:
    df = univ.copy()
    df["strike"] = [_strike(t) for t in df["ticker"]]

    # --- contract-level quality gates ------------------------------------
    df = df[(df["price"] >= PRICE_FLOOR) & (df["price"] <= PRICE_CEIL)]
    df = df[df["volume"] > 0]

    # --- event-level gates ------------------------------------------------
    g = df.groupby("event_ticker")
    size = g.size()
    keep = set(size[(size >= MIN_EVENT_SIZE) & (size <= MAX_EVENT_SIZE)].index)
    df = df[df["event_ticker"].isin(keep)]

    # no-arbitrage coherence check on mutually-exclusive events
    mx = df[df["mutually_exclusive"]]
    if len(mx):
        psum = mx.groupby("event_ticker")["price"].sum()
        bad = set(psum[(psum - 1.0).abs() > COHERENCE_TOL].index)
        df = df[~df["event_ticker"].isin(bad)]

    # --- rank events by liquidity and fill to the target ------------------
    vol = df.groupby("event_ticker")["volume"].sum().sort_values(ascending=False)

    picked, total = [], 0
    for ev in vol.index:
        block = df[df["event_ticker"] == ev]
        if total + len(block) > n_target:
            continue
        picked.append(block)
        total += len(block)
        if total == n_target:
            break

    out = pd.concat(picked)

    # --- annotate ---------------------------------------------------------
    parts = []
    for ev, grp in out.groupby("event_ticker"):
        grp = grp.sort_values("strike" if grp["strike"].notna().all() else "ticker")
        grp = grp.copy()
        grp["event_size"] = len(grp)
        grp["event_price_sum"] = round(float(grp["price"].sum()), 4)
        grp["event_volume"] = float(grp["volume"].sum())
        grp["structure"] = _classify(grp)
        parts.append(grp)

    out = pd.concat(parts)
    out = out.sort_values(["structure", "event_volume", "event_ticker", "strike"],
                          ascending=[True, False, True, True])

    cols = ["ticker", "name", "category", "series_ticker", "event_ticker",
            "structure", "mutually_exclusive", "event_size", "event_price_sum",
            "strike", "price", "volume", "event_volume"]
    return out[cols].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=200, help="number of contracts")
    ap.add_argument("--out", default="contracts_200.csv")
    ap.add_argument("--refetch", action="store_true",
                    help="re-pull from the Kalshi API instead of using the cache")
    a = ap.parse_args()

    univ = load_universe(use_cache=not a.refetch)
    sel = select(univ, n_target=a.n)
    sel.to_csv(a.out, index=False)

    print(f"\n  wrote {len(sel)} contracts to {a.out}")
    print(f"  events            : {sel['event_ticker'].nunique()}")
    print(f"  price range       : {sel['price'].min():.2f} - {sel['price'].max():.2f}")
    print(f"  total volume      : {sel['volume'].sum():,.0f}")

    print("\n  by structure (covariance form implied):")
    for s, grp in sel.groupby("structure"):
        print(f"    {s:14s} {len(grp):4d} contracts  "
              f"{grp['event_ticker'].nunique():3d} events")

    print("\n  by category:")
    for c, k in sel["category"].value_counts().items():
        print(f"    {k:4d}  {c}")

    mx = sel[sel["mutually_exclusive"]]
    if len(mx):
        sums = mx.groupby("event_ticker")["event_price_sum"].first()
        print(f"\n  exclusive-event price sums: min={sums.min():.3f} "
              f"median={sums.median():.3f} max={sums.max():.3f}")
    print()


if __name__ == "__main__":
    main()
