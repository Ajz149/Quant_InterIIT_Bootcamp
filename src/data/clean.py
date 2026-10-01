"""Shared cleaning: data/raw/*.parquet -> regular grid in data/interim/*_clean.parquet.

Fixes structure (gaps, flags) but never edits real market values.
Outliers are flagged and reported, not removed.
"""
from pathlib import Path

import numpy as np
import pandas as pd

FREQ = {"1h": "1h", "1d": "1D"}
SHOCK_RET = {"1h": 0.10, "1d": 0.25}      # |log return| above this is reported
ZERO_FILL = ["volume", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"]


def clean(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Return a gap-free, flagged OHLCV frame for one symbol/interval."""
    assert str(df.index.tz) == "UTC", "index must be UTC"
    assert df.index.is_monotonic_increasing and not df.index.duplicated().any()

    grid = pd.date_range(df.index.min(), df.index.max(), freq=FREQ[interval])
    out = df.drop(columns=["close_time"], errors="ignore").reindex(grid)
    out.index.name = "timestamp"

    out["is_filled"] = out["close"].isna()
    out["close"] = out["close"].ffill()
    for c in ["open", "high", "low"]:
        out[c] = out[c].fillna(out["close"])
    out[ZERO_FILL] = out[ZERO_FILL].fillna(0.0)

    out["is_zero_vol"] = (out["volume"] == 0) & ~out["is_filled"]
    return out


def report(df: pd.DataFrame, interval: str, name: str) -> None:
    ret = np.log(df["close"]).diff()
    shocks = ret[ret.abs() > SHOCK_RET[interval]]
    wick = (df["high"] - df["low"]) / df["close"]
    print(f"\n=== {name} ===")
    print(f"rows            : {len(df)}")
    print(f"filled bars     : {int(df['is_filled'].sum())}")
    print(f"zero-vol bars   : {int(df['is_zero_vol'].sum())}")
    print(f"NaNs            : {int(df.isna().sum().sum())}")
    print(f"shock bars (>{SHOCK_RET[interval]:.0%}): {len(shocks)}")
    for ts, r in shocks.head(10).items():
        print(f"   {ts}  ret={r:+.2%}")
    print(f"max range/close : {wick.max():.2%} at {wick.idxmax()}")


def main() -> None:
    raw, out_dir = Path("data/raw"), Path("data/interim")
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(raw.glob("*_*.parquet"))
    if not files:
        raise SystemExit("No raw parquet files found. Run src.data.acquire first.")
    for f in files:
        symbol, interval = f.stem.split("_")
        df = clean(pd.read_parquet(f), interval)
        assert not df.isna().any().any(), "NaNs remain after cleaning"
        out = out_dir / f"{symbol}_{interval}_clean.parquet"
        df.to_parquet(out)
        report(df, interval, out.name)


if __name__ == "__main__":
    main()