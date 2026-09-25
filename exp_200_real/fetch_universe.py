# fetch_universe.py - bulk-fetch the open Kalshi universe for scaling experiments.
#
# Unlike kalshi_data.py (12 hand-picked tickers), this walks the /events
# endpoint with nested markets so that we keep the *event structure*, which is
# what carries the exclusivity information.  Kalshi exposes a per-event
# `mutually_exclusive` flag, so exclusivity is taken from the exchange rather
# than inferred from a price heuristic.

import time
import requests
import pandas as pd
from typing import Optional

BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
CACHE = "kalshi_universe.csv"


def _price(m: dict) -> float:
    """last traded price, else bid/ask midpoint, else 0 (unusable)"""
    def f(key):
        try:
            return float(m.get(key) or 0)
        except (TypeError, ValueError):
            return 0.0

    p = f("last_price_dollars")
    if 0.0 < p < 1.0:
        return p
    bid, ask = f("yes_bid_dollars"), f("yes_ask_dollars")
    if bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    return 0.0


def fetch_universe(max_pages: int = 40, page_size: int = 200,
                   verbose: bool = True) -> pd.DataFrame:
    """walk /events?with_nested_markets and flatten to one row per contract"""
    rows = []
    cursor: Optional[str] = None
    pages = 0

    while pages < max_pages:
        params = {"status": "open", "limit": page_size,
                  "with_nested_markets": "true"}
        if cursor:
            params["cursor"] = cursor

        r = requests.get(f"{BASE_URL}/events", params=params, timeout=60)
        r.raise_for_status()
        payload = r.json()
        events = payload.get("events", [])
        if not events:
            break

        for e in events:
            mx = bool(e.get("mutually_exclusive"))
            for m in e.get("markets", []) or []:
                p = _price(m)
                if not (0.0 < p < 1.0):
                    continue  # untraded / no two-sided quote
                try:
                    vol = float(m.get("volume_fp") or 0)
                except (TypeError, ValueError):
                    vol = 0.0
                rows.append({
                    "ticker": m.get("ticker", ""),
                    "name": (m.get("title") or "")[:180],
                    "category": e.get("category", "Other"),
                    "series_ticker": e.get("series_ticker", ""),
                    "event_ticker": e.get("event_ticker", ""),
                    "mutually_exclusive": mx,
                    "price": p,
                    "volume": vol,
                })

        cursor = payload.get("cursor")
        pages += 1
        if verbose:
            print(f"    page {pages:3d}  cumulative contracts: {len(rows)}")
        if not cursor:
            break
        time.sleep(0.05)  # be polite

    df = pd.DataFrame(rows).drop_duplicates(subset="ticker").reset_index(drop=True)
    return df


def load_universe(use_cache: bool = True, **kw) -> pd.DataFrame:
    if use_cache:
        try:
            df = pd.read_csv(CACHE)
            print(f"  Loaded {len(df)} contracts from {CACHE}")
            return df
        except FileNotFoundError:
            pass
    df = fetch_universe(**kw)
    df.to_csv(CACHE, index=False)
    print(f"  Cached {len(df)} contracts to {CACHE}")
    return df


if __name__ == "__main__":
    print("\n=== KALSHI UNIVERSE FETCH ===\n")
    df = load_universe(use_cache=False)

    print(f"\n  contracts        : {len(df)}")
    print(f"  distinct events  : {df['event_ticker'].nunique()}")
    print(f"  distinct series  : {df['series_ticker'].nunique()}")

    mx_events = df[df["mutually_exclusive"]]["event_ticker"].nunique()
    print(f"  mutually-exclusive events : {mx_events}")
    print(f"  contracts inside them     : {int(df['mutually_exclusive'].sum())}")

    print("\n  by category:")
    for cat, k in df["category"].value_counts().items():
        print(f"    {k:6d}  {cat}")

    sizes = df.groupby("event_ticker").size()
    print(f"\n  contracts per event: mean={sizes.mean():.1f} "
          f"median={sizes.median():.0f} max={sizes.max()}")
    print()
