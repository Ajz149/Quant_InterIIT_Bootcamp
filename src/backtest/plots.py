"""Equity curve, drawdown and trade history plots."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # save to file, no pop-up windows
import matplotlib.pyplot as plt

from src.backtest.metrics import buy_and_hold


def plot_report(df, res, title: str, path: str) -> None:
    bh = buy_and_hold(df, res.initial_capital)
    dd = (res.equity / res.equity.cummax() - 1) * 100
    bh_dd = (bh / bh.cummax() - 1) * 100
    fig, ax = plt.subplots(3, 1, figsize=(13, 11), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1.5, 3]})
    ax[0].plot(res.equity, lw=1.2, label="Strategy")
    ax[0].plot(bh, lw=1, alpha=0.7, label="Buy & hold")
    ax[0].set_ylabel("Equity (USDT)"); ax[0].set_title(title); ax[0].legend()

    ax[1].fill_between(dd.index, dd, 0, color="tab:red", alpha=0.4, label="Strategy")
    ax[1].plot(bh_dd, color="grey", lw=0.8, label="Buy & hold")
    ax[1].set_ylabel("Drawdown (%)"); ax[1].legend()

    ax[2].plot(df["close"], color="grey", lw=0.6)
    tr = res.trades
    if len(tr):
        ax[2].scatter(tr["entry_time"], tr["entry_px"], marker="^", s=12,
                      c="tab:green", label="Entry")
        ax[2].scatter(tr["exit_time"], tr["exit_px"], marker="v", s=12,
                      c="tab:red", label="Exit")
        ax[2].legend()
    ax[2].set_ylabel("Price")

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)