"""Performance metrics: the 15 required by the PS plus a few extras."""
import numpy as np
import pandas as pd

REQUIRED_KEYS = [
    "gross_profit", "net_profit", "total_closed_trades", "win_rate_pct",
    "max_drawdown_pct", "gross_loss", "avg_winning_trade", "avg_losing_trade",
    "buy_hold_return_pct", "largest_losing_trade", "largest_winning_trade",
    "sharpe_ratio", "sortino_ratio", "avg_hold_hours", "max_hold_hours"]


def buy_and_hold(df: pd.DataFrame, initial_capital: float) -> pd.Series:
    """Benchmark equity: buy at the first open, hold, no fees."""
    return initial_capital / df["open"].iloc[0] * df["close"]


def _with_start(eq: pd.Series, init: float) -> pd.Series:
    start = pd.Series([init], index=[eq.index[0] - pd.Timedelta(hours=1)])
    return pd.concat([start, eq])


def quarterly_returns(eq0: pd.Series) -> pd.Series:
    key = eq0.index.tz_convert(None).to_period("Q")
    q = eq0.groupby(key).last()
    prev = q.shift(1)
    prev.iloc[0] = eq0.iloc[0]
    return q / prev - 1


def quarterly_table(res, df: pd.DataFrame) -> pd.DataFrame:
    init = res.initial_capital
    s = quarterly_returns(_with_start(res.equity, init))
    b = quarterly_returns(_with_start(buy_and_hold(df, init), init))
    q = pd.DataFrame({"strategy_pct": s * 100, "buy_hold_pct": b * 100})
    q["beat"] = q["strategy_pct"] > q["buy_hold_pct"]
    return q


def compute_metrics(res, df: pd.DataFrame) -> dict:
    init = res.initial_capital
    eq0 = _with_start(res.equity, init)
    bh0 = _with_start(buy_and_hold(df, init), init)
    tr = res.trades
    net = tr["net_pnl"].astype(float)
    wins, losses = net[net > 0], net[net < 0]
    hold = ((tr["exit_time"] - tr["entry_time"]).dt.total_seconds() / 3600
            if len(tr) else pd.Series(dtype=float))

    dd = eq0 / eq0.cummax() - 1
    under = dd < 0
    dd_days = under.groupby((~under).cumsum()).sum().max() / 24

    r = eq0.resample("1D").last().pct_change().dropna()          # daily returns
    sd = r.std()
    dsd = np.sqrt((np.minimum(r, 0) ** 2).mean())
    sharpe = r.mean() / sd * np.sqrt(365) if sd > 0 else np.nan
    sortino = r.mean() / dsd * np.sqrt(365) if dsd > 0 else np.nan

    total = res.equity.iloc[-1] / init
    years = len(res.equity) / (24 * 365)
    ann = (total ** (1 / years) - 1) * 100 if total > 0 else np.nan
    mdd = -dd.min() * 100
    q = quarterly_table(res, df)

    m = {
        # ---- the 15 metrics required by the PS ----
        "gross_profit": wins.sum(),
        "net_profit": res.equity.iloc[-1] - init,
        "total_closed_trades": int(len(tr)),
        "win_rate_pct": 100 * (net > 0).mean() if len(tr) else 0.0,
        "max_drawdown_pct": mdd,
        "gross_loss": losses.sum(),
        "avg_winning_trade": wins.mean() if len(wins) else 0.0,
        "avg_losing_trade": losses.mean() if len(losses) else 0.0,
        "buy_hold_return_pct": (bh0.iloc[-1] / init - 1) * 100,
        "largest_losing_trade": losses.min() if len(losses) else 0.0,
        "largest_winning_trade": wins.max() if len(wins) else 0.0,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "avg_hold_hours": hold.mean() if len(hold) else 0.0,
        "max_hold_hours": hold.max() if len(hold) else 0.0,
        # ---- extras ----
        "total_return_pct": (total - 1) * 100,
        "annualized_return_pct": ann,
        "calmar_ratio": ann / mdd if mdd > 0 else np.nan,
        "max_drawdown_usdt": (eq0.cummax() - eq0).max(),
        "max_underwater_days": dd_days,
        "profit_factor": wins.sum() / abs(losses.sum()) if len(losses) else np.nan,
        "fees_paid": tr["fees"].sum() if len(tr) else 0.0,
        "exposure_pct": res.position.ne(0).mean() * 100,
        "n_quarters": int(len(q)),
        "quarters_beating_bh_pct": 100 * q["beat"].mean(),
    }
    return {k: float(v) if not isinstance(v, int) else v for k, v in m.items()}


def format_metrics(m: dict) -> str:
    return "\n".join(f"{k:<26}{v:>14,.2f}" for k, v in m.items())