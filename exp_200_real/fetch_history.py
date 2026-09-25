# fetch_history.py - daily price history for a contract set, from Kalshi
# candlesticks.  This is the raw material for estimating correlations
# empirically instead of asserting them.

import time
import argparse
import requests
import pandas as pd

BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
CACHE = "price_history.csv"
DAY = 1440  # period_interval in minutes


def fetch_one(series: str, ticker: str, days: int = 90, timeout: int = 30):
    """daily candles for a single contract"""
    now = int(time.time())
    url = f"{BASE_URL}/series/{series}/markets/{ticker}/candlesticks"
    r = requests.get(url, params={"start_ts": now - days * 86400,
                                  "end_ts": now,
                                  "period_interval": DAY}, timeout=timeout)
    r.raise_for_status()

    out = []
    for c in r.json().get("candlesticks", []):
        px = c.get("price") or {}

        def f(d, key):
            try:
                return float(d.get(key) or "nan")
            except (TypeError, ValueError):
                return float("nan")

        close = f(px, "close_dollars")
        if close != close:                      # NaN -> fall back to the mean
            close = f(px, "mean_dollars")
        try:
            vol = float(c.get("volume_fp") or 0)
        except (TypeError, ValueError):
            vol = 0.0

        out.append({"ticker": ticker,
                    "ts": int(c.get("end_period_ts", 0)),
                    "close": close,
                    "volume": vol})
    return out


def fetch_history(contracts: pd.DataFrame, days: int = 90,
                  pause: float = 0.08) -> pd.DataFrame:
    rows, failed = [], []
    n = len(contracts)
    for i, (_, c) in enumerate(contracts.iterrows(), 1):
        try:
            got = fetch_one(c["series_ticker"], c["ticker"], days=days)
            rows.extend(got)
            status = f"{len(got):3d} candles"
        except Exception as exc:
            failed.append(c["ticker"])
            status = f"FAILED {type(exc).__name__}"
        if i % 25 == 0 or i == n:
            print(f"    [{i:4d}/{n}] {c['ticker'][:44]:44s} {status}")
        time.sleep(pause)

    df = pd.DataFrame(rows)
    if len(df):
        df["date"] = pd.to_datetime(df["ts"], unit="s").dt.date
    if failed:
        print(f"\n  {len(failed)} contracts returned no history")
    return df


def load_history(contracts_csv="contracts_200.csv", days=90,
                 use_cache=True) -> pd.DataFrame:
    if use_cache:
        try:
            return pd.read_csv(CACHE)
        except FileNotFoundError:
            pass
    contracts = pd.read_csv(contracts_csv)
    df = fetch_history(contracts, days=days)
    df.to_csv(CACHE, index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--contracts", default="contracts_200.csv")
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--out", default=CACHE)
    a = ap.parse_args()

    contracts = pd.read_csv(a.contracts)
    print(f"\n=== PRICE HISTORY: {len(contracts)} contracts, {a.days} days ===\n")
    df = fetch_history(contracts, days=a.days)
    df.to_csv(a.out, index=False)

    print(f"\n  wrote {len(df):,} rows to {a.out}")
    print(f"  contracts with history : {df['ticker'].nunique()} / {len(contracts)}")
    if len(df):
        per = df.groupby("ticker").size()
        print(f"  candles per contract   : min={per.min()} median={per.median():.0f} max={per.max()}")
        print(f"  date range             : {df['date'].min()} .. {df['date'].max()}")
        print(f"  rows with a close price: {df['close'].notna().sum():,}")
    print()
