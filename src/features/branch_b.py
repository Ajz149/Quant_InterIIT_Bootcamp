"""Branch B: rich ML feature set = Branch A + returns, candles, microstructure,
regime, cross-asset and time features.

Usage:  python -m src.features.branch_b
Reads  data/interim/{SYMBOL}_1h_clean.parquet  (both symbols, for cross-asset)
Writes data/processed/{SYMBOL}_1h_B.parquet
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.features import indicators as ind
from src.features.branch_a import build_features_a

WARMUP = 800  # longest chain: 24-bar vol -> 720-bar percentile


def _pct_rank(s: pd.Series, window: int) -> pd.Series:
    """Percentile of the latest value within its trailing window (causal)."""
    return s.rolling(window).apply(lambda x: (x[-1] >= x).mean(), raw=True)


def build_features_b(df: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """Branch B features for `df`; `other` is the other coin's clean frame."""
    other = other.reindex(df.index)
    o, h, l, c, v = (df[k] for k in ["open", "high", "low", "close", "volume"])
    oc = other["close"]
    f = build_features_a(df)

    # --- multi-horizon returns -------------------------------------------
    for n in (3, 6, 12, 48, 168):
        f[f"ret_{n}"] = ind.log_return(c, n)

    # --- candle shape ------------------------------------------------------
    top = pd.concat([o, c], axis=1).max(axis=1)
    bot = pd.concat([o, c], axis=1).min(axis=1)
    f["range_pct"] = (h - l) / c
    f["body_pct"] = (c - o) / c
    f["upper_wick"] = (h - top) / c
    f["lower_wick"] = (bot - l) / c

    # --- microstructure ----------------------------------------------------
    tb = (df["taker_buy_base"] / v.where(v > 0)).clip(0, 1)
    f["tb_ratio"] = tb.fillna(0.5)                      # neutral on empty bars
    f["tb_ratio_24"] = f["tb_ratio"].rolling(24).mean()

    size = v / df["trades"].where(df["trades"] > 0)     # avg trade size
    ratio = size / size.rolling(168, min_periods=100).mean()
    f["size_ratio_168"] = ratio.where(size.notna(), 1.0)

    ami = (ind.log_return(c, 1).abs().rolling(24).mean()
           / df["quote_volume"].rolling(24).mean().replace(0, np.nan))
    lam = np.log(ami.replace(0, np.nan))
    f["amihud_rel"] = lam - lam.rolling(336, min_periods=200).mean()

    # --- regime ------------------------------------------------------------
    f["vol_ratio_24_168"] = f["vol_24"] / f["vol_168"]
    f["vol_pct_720"] = _pct_rank(f["vol_24"], 720)
    e50, e200 = ind.ema(c, 50), ind.ema(c, 200)
    f["dist_ema200"] = c / e200 - 1
    f["ema_spread"] = (e50 - e200) / c
    f["er_24"] = ind.efficiency_ratio(c, 24)
    f["er_168"] = ind.efficiency_ratio(c, 168)
    adx, pdi, mdi = ind.adx(h, l, c, 14)
    f["adx_14"] = adx
    f["di_diff"] = pdi - mdi

    # --- cross-asset ---------------------------------------------------------
    r1, xr1 = ind.log_return(c, 1), ind.log_return(oc, 1)
    f["x_ret_1"] = xr1
    f["x_ret_24"] = ind.log_return(oc, 24)
    f["x_vol_24"] = ind.realized_vol(oc, 24)
    f["corr_168"] = r1.rolling(168).corr(xr1)
    f["rel_ret_24"] = f["ret_24"] - f["x_ret_24"]
    f["spread_z_168"] = ind.zscore(np.log(c) - np.log(oc), 168)

    # --- time (deterministic from the timestamp, so no leakage) -------------
    hr, dow = df.index.hour, df.index.dayofweek
    f["hour_sin"] = np.sin(2 * np.pi * hr / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    f["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    f["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    return f.replace([np.inf, -np.inf], np.nan)


def main() -> None:
    cfg = yaml.safe_load(open("configs/base.yaml", encoding="utf-8-sig"))["data"]
    syms = cfg["symbols"]
    clean = {s: pd.read_parquet(f"data/interim/{s}_1h_clean.parquet") for s in syms}
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    for sym, oth in zip(syms, syms[::-1]):          # BTC<->ETH
        feats = build_features_b(clean[sym], clean[oth])
        base = clean[sym][["open", "high", "low", "close", "volume",
                           "is_filled", "is_zero_vol"]]
        out = pd.concat([base, feats], axis=1).iloc[WARMUP:]

        assert not out.isna().any().any(), "NaN left after warmup"
        assert np.isfinite(out[feats.columns]).all().all(), "inf in features"

        path = out_dir / f"{sym}_1h_B.parquet"
        out.to_parquet(path)
        new = [c for c in feats.columns if c not in build_features_a(clean[sym]).columns]
        print(f"\n=== {path.name} ===  rows={len(out)}  features={feats.shape[1]}")
        print(out[new].describe().T[["mean", "std", "min", "max"]].round(4))


if __name__ == "__main__":
    main()