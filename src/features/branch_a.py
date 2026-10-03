"""Branch A: compact statistical feature set.

Usage:  python -m src.features.branch_a
Reads  data/interim/{SYMBOL}_1h_clean.parquet
Writes data/processed/{SYMBOL}_1h_A.parquet
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.features import indicators as ind

WARMUP = 200  # rows dropped at the start so slow indicators have converged


def build_features_a(df: pd.DataFrame) -> pd.DataFrame:
    """Return the Branch A feature frame (same index as df, NaN during warmup)."""
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    f = pd.DataFrame(index=df.index)

    f["ret_1"] = ind.log_return(c, 1)
    f["ret_24"] = ind.log_return(c, 24)
    f["vol_24"] = ind.realized_vol(c, 24)
    f["vol_168"] = ind.realized_vol(c, 168)
    f["z_24"] = ind.zscore(c, 24)
    f["z_168"] = ind.zscore(c, 168)
    f["atr_pct"] = ind.atr(h, l, c, 14) / c
    f["rsi_14"] = ind.rsi(c, 14)

    line, _, hist = ind.macd(c)
    f["macd_pct"] = line / c
    f["macd_hist_pct"] = hist / c

    f["bb_pctb"], f["bb_width"] = ind.bollinger(c, 20, 2.0)
    f["vol_ratio_24"] = v / v.rolling(24).mean().replace(0, np.nan)

    return f.replace([np.inf, -np.inf], np.nan)


def main() -> None:
    cfg = yaml.safe_load(open("configs/base.yaml", encoding="utf-8-sig"))["data"]
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    for symbol in cfg["symbols"]:
        clean = pd.read_parquet(f"data/interim/{symbol}_1h_clean.parquet")
        feats = build_features_a(clean)

        # keep prices + flags next to features: the backtester needs open/close
        base = clean[["open", "high", "low", "close", "volume", "is_filled", "is_zero_vol"]]
        out = pd.concat([base, feats], axis=1).iloc[WARMUP:]

        assert not out.isna().any().any(), "NaN left after warmup"
        assert np.isfinite(out[feats.columns]).all().all(), "inf in features"

        path = out_dir / f"{symbol}_1h_A.parquet"
        out.to_parquet(path)
        print(f"\n=== {path.name} ===  rows={len(out)}  features={feats.shape[1]}")
        print(out[feats.columns].describe().T[["mean", "std", "min", "max"]].round(4))


if __name__ == "__main__":
    main()