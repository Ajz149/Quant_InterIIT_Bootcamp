"""Custom event-driven backtester (no external libraries).

Timing contract
  * signal[t] is the TARGET position decided at the CLOSE of bar t (+1/0/-1)
  * it is filled at the OPEN of bar t+1, paying `fee` on the traded notional
  * optional stop-loss / take-profit are checked intrabar from the entry bar on;
    if both are touched in one bar the stop is assumed to hit first
  * synthetic (is_filled) bars are never traded on: orders wait for the next real bar
  * after a stop/target exit, no re-entry in the same direction until the signal changes
  * a position still open on the last bar is closed at that bar's close ('eod')
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADE_COLS = ["entry_time", "exit_time", "direction", "entry_px", "exit_px", "units",
              "gross_pnl", "fees", "net_pnl", "ret_pct", "bars_held", "exit_reason"]


@dataclass
class BacktestResult:
    equity: pd.Series        # mark-to-market at every bar close
    trades: pd.DataFrame     # one row per closed trade
    position: pd.Series      # direction held during each bar (+1/0/-1)
    initial_capital: float
    fee: float


def run_backtest(df: pd.DataFrame, signal: pd.Series, fee: float = 0.0015,
                 initial_capital: float = 10_000.0, allow_short: bool = False,
                 sl_pct: pd.Series | None = None, tp_pct: pd.Series | None = None,
                 size: pd.Series | None = None) -> BacktestResult:
    """Run one strategy. sl_pct / tp_pct are distances as a fraction of the entry price
    (e.g. 0.03 = 3%), size is the fraction of equity to deploy (0..1). All three are read
    at the signal bar, i.e. they must only use information known at that close."""
    idx, n = df.index, len(df)
    O, H, L, C = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    filled = df["is_filled"].to_numpy(bool) if "is_filled" in df else np.zeros(n, bool)

    def aligned(s, default):
        return np.full(n, default) if s is None else s.reindex(idx).to_numpy(float)

    sig = np.sign(np.nan_to_num(aligned(signal, 0.0)))
    if not allow_short:
        sig = np.maximum(sig, 0.0)
    sl_arr, tp_arr = aligned(sl_pct, np.nan), aligned(tp_pct, np.nan)
    sz_arr = np.clip(np.nan_to_num(aligned(size, 1.0)), 0.0, 1.0)

    cash, units, d, lock = float(initial_capital), 0.0, 0, 0
    pend, p_sl, p_tp, p_sz = 0, np.nan, np.nan, 1.0
    sl_lvl = tp_lvl = np.nan          # NaN levels never trigger (comparisons are False)
    pos: dict = {}
    trades: list = []
    equity, held = np.empty(n), np.zeros(n)

    def enter_trade(px, i, direction, sz, sl_p, tp_p):
        nonlocal cash, units, d, sl_lvl, tp_lvl
        u = cash * sz / (px * (1 + fee))              # outlay incl. fee = cash * sz
        pos.update(i=i, px=px, dir=direction, cash0=cash)
        units = direction * u
        cash -= units * px + fee * u * px
        d = direction
        sl_lvl = px * (1 - direction * sl_p)
        tp_lvl = px * (1 + direction * tp_p)

    def exit_trade(px, i, reason):
        nonlocal cash, units, d
        u = abs(units)
        cash += units * px - fee * u * px
        net = cash - pos["cash0"]
        fees = fee * u * (pos["px"] + px)
        trades.append({
            "entry_time": idx[pos["i"]], "exit_time": idx[i], "direction": pos["dir"],
            "entry_px": pos["px"], "exit_px": px, "units": u,
            "gross_pnl": net + fees, "fees": fees, "net_pnl": net,
            "ret_pct": 100 * net / (u * pos["px"]), "bars_held": i - pos["i"],
            "exit_reason": reason})
        units, d = 0.0, 0

    for i in range(n):
        o, h, l, c = O[i], H[i], L[i], C[i]
        real = not filled[i]

        # 1) execute the order decided at the previous close, at this bar's open
        if real and pend != d:
            if d != 0:
                exit_trade(o, i, "signal")
            if pend != 0 and p_sz > 0:
                enter_trade(o, i, pend, p_sz, p_sl, p_tp)
        held[i] = d

        # 2) intrabar stop / target (stop first)
        if real and d != 0:
            if d == 1:
                if l <= sl_lvl:
                    exit_trade(min(o, sl_lvl), i, "stop"); lock = 1
                elif h >= tp_lvl:
                    exit_trade(max(o, tp_lvl), i, "target"); lock = 1
            else:
                if h >= sl_lvl:
                    exit_trade(max(o, sl_lvl), i, "stop"); lock = -1
                elif l <= tp_lvl:
                    exit_trade(min(o, tp_lvl), i, "target"); lock = -1

        equity[i] = cash + units * c

        # 3) decide at this bar's close, to be filled at the next open
        s = sig[i]
        if lock != 0 and s != lock:
            lock = 0
        pend = 0 if lock != 0 else int(s)
        p_sl, p_tp, p_sz = sl_arr[i], tp_arr[i], sz_arr[i]

    if d != 0:                                         # close whatever is still open
        exit_trade(C[-1], n - 1, "eod")
        equity[-1] = cash

    return BacktestResult(pd.Series(equity, idx, name="equity"),
                          pd.DataFrame(trades, columns=TRADE_COLS),
                          pd.Series(held, idx, name="position"),
                          float(initial_capital), fee)