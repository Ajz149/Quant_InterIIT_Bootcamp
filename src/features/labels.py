"""Training targets. These use FUTURE data on purpose: never use them as features.

Convention (matches the backtest engine): decide at close of bar t, fill at the
open of bar t+1, and the trade lives through bars t+1 ... t+horizon.
"""
import numpy as np
import pandas as pd

from src.features import indicators as ind


def forward_return(df: pd.DataFrame, horizon: int) -> pd.Series:
    """log(open[t+1+h] / open[t+1])."""
    entry = df["open"].shift(-1)
    exit_ = df["open"].shift(-(horizon + 1))
    return np.log(exit_ / entry).rename("fwd_ret")


def triple_barrier(df: pd.DataFrame, horizon: int = 24, k_up: float = 3.0,
                   k_dn: float = 3.0, atr_n: int = 14) -> pd.DataFrame:
    """Label each bar t with +1 / -1 / 0 (take-profit / stop-loss / time-out)."""
    c = df["close"]
    width = (ind.atr(df["high"], df["low"], c, atr_n) / c).to_numpy()  # known at t
    O, H, L, C = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    n = len(df)

    label = np.full(n, np.nan)
    ret = np.full(n, np.nan)
    hold = np.full(n, np.nan)

    for t in range(n - horizon - 1):
        if np.isnan(width[t]):
            continue
        entry = O[t + 1]
        up = entry * (1 + k_up * width[t])
        dn = entry * (1 - k_dn * width[t])
        for j in range(t + 1, t + horizon + 1):
            if L[j] <= dn:                       # stop checked first (conservative)
                label[t], px, hold[t] = -1, dn, j - t
                break
            if H[j] >= up:
                label[t], px, hold[t] = 1, up, j - t
                break
        else:                                    # no barrier touched: time-out
            label[t], px, hold[t] = 0, C[t + horizon], horizon
        ret[t] = np.log(px / entry)

    return pd.DataFrame({"tb_label": label, "tb_ret": ret, "tb_hold": hold},
                        index=df.index)