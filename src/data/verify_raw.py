"""Sanity-check raw parquet files. Read-only."""
import sys
from pathlib import Path

import pandas as pd

FREQ = {"1h": "1h", "1d": "1D", "15m": "15min", "4h": "4h"}


def check(path: Path):
    interval = path.stem.split("_")[1]
    df = pd.read_parquet(path)
    expected = pd.date_range(df.index.min(), df.index.max(), freq=FREQ[interval])
    missing = expected.difference(df.index)
    print(f"\n=== {path.name} ===")
    print(f"rows           : {len(df)}  (expected {len(expected)})")
    print(f"range          : {df.index.min()} -> {df.index.max()}")
    print(f"missing bars   : {len(missing)}")
    if len(missing):
        print("  first few    :", list(missing[:5]))
    print(f"duplicates     : {df.index.duplicated().sum()}")
    print(f"NaNs           : {int(df.isna().sum().sum())}")
    print(f"zero-volume    : {(df['volume'] == 0).sum()}")
    bad = ((df["high"] < df[["open", "close"]].max(axis=1)) |
           (df["low"] > df[["open", "close"]].min(axis=1)) |
           (df["low"] <= 0)).sum()
    print(f"bad OHLC bars  : {bad}")
    print(f"close min/max  : {df['close'].min():.2f} / {df['close'].max():.2f}")
    print(f"buy&hold       : {df['close'].iloc[-1] / df['close'].iloc[0] - 1:.1%}")


if __name__ == "__main__":
    files = sorted(Path("data/raw").glob("*.parquet"))
    if not files:
        sys.exit("No parquet files in data/raw")
    for f in files:
        check(f)