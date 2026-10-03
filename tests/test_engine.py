import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import run_backtest
from src.backtest.metrics import REQUIRED_KEYS, compute_metrics

FEE, CAP = 0.0015, 10_000.0


def frame(n=20, price=100.0):
    idx = pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"open": price, "high": price, "low": price,
                         "close": price, "is_filled": False}, index=idx)


def sig(df, start, stop=None, val=1.0):
    s = pd.Series(0.0, index=df.index)
    s.iloc[start:stop] = val
    return s


def test_flat_signal_means_no_trades_and_no_pnl():
    df = frame()
    res = run_backtest(df, sig(df, 0, 0), FEE, CAP)
    assert len(res.trades) == 0
    assert (res.equity == CAP).all()


def test_fills_at_next_open_not_signal_close():
    df = frame()
    df.loc[df.index[5], ["open", "high"]] = 105.0
    res = run_backtest(df, sig(df, 4, 5), FEE, CAP)        # signal at bar 4 only
    t = res.trades.iloc[0]
    assert t["entry_time"] == df.index[5] and t["entry_px"] == 105.0
    assert t["bars_held"] == 1


def test_round_trip_fees_on_flat_price():
    df = frame()
    res = run_backtest(df, sig(df, 2, 5), FEE, CAP)
    u = CAP / (100 * (1 + FEE))
    assert res.equity.iloc[-1] == pytest.approx(CAP * (1 - FEE) / (1 + FEE))
    t = res.trades.iloc[0]
    assert t["fees"] == pytest.approx(FEE * u * 200)
    assert t["bars_held"] == 3


def test_stop_loss_hit_and_no_instant_reentry():
    df = frame()
    df.iloc[6, df.columns.get_loc("low")] = 90.0
    sl = pd.Series(0.05, index=df.index)
    res = run_backtest(df, sig(df, 2), FEE, CAP, sl_pct=sl)  # signal stays long
    assert len(res.trades) == 1                              # lockout: no re-buy
    t = res.trades.iloc[0]
    assert t["exit_reason"] == "stop" and t["exit_px"] == pytest.approx(95.0)
    assert t["exit_time"] == df.index[6]


def test_stop_wins_when_stop_and_target_in_same_bar():
    df = frame()
    df.iloc[6, df.columns.get_loc("high")] = 110.0
    df.iloc[6, df.columns.get_loc("low")] = 90.0
    s = pd.Series(0.05, index=df.index)
    res = run_backtest(df, sig(df, 2), FEE, CAP, sl_pct=s, tp_pct=s)
    assert res.trades.iloc[0]["exit_reason"] == "stop"


def test_gap_through_stop_fills_at_open():
    df = frame()
    df.loc[df.index[6], ["open", "high", "low", "close"]] = [90.0, 91.0, 89.0, 90.0]
    sl = pd.Series(0.05, index=df.index)
    res = run_backtest(df, sig(df, 2), FEE, CAP, sl_pct=sl)
    assert res.trades.iloc[0]["exit_px"] == pytest.approx(90.0)


def test_filled_bar_is_never_traded():
    df = frame()
    df.loc[df.index[3], "is_filled"] = True
    res = run_backtest(df, sig(df, 2, 8), FEE, CAP)
    assert res.trades.iloc[0]["entry_time"] == df.index[4]   # delayed past the gap


def test_open_position_closed_at_end_and_pnl_reconciles():
    p = 100 + np.arange(20.0)
    idx = pd.date_range("2021-01-01", periods=20, freq="1h", tz="UTC")
    df = pd.DataFrame({"open": p, "high": p, "low": p, "close": p, "is_filled": False}, index=idx)
    res = run_backtest(df, sig(df, 1), FEE, CAP)
    t = res.trades.iloc[0]
    assert len(res.trades) == 1 and t["exit_reason"] == "eod"
    assert t["exit_time"] == df.index[-1]
    assert res.equity.iloc[-1] == pytest.approx(CAP + res.trades["net_pnl"].sum())


def test_short_only_when_allowed():
    p = 100 - np.arange(20.0)
    idx = pd.date_range("2021-01-01", periods=20, freq="1h", tz="UTC")
    df = pd.DataFrame({"open": p, "high": p, "low": p, "close": p, "is_filled": False}, index=idx)
    s = sig(df, 1, None, -1.0)
    assert len(run_backtest(df, s, FEE, CAP, allow_short=False).trades) == 0
    res = run_backtest(df, s, FEE, CAP, allow_short=True)
    assert res.trades.iloc[0]["direction"] == -1 and res.trades.iloc[0]["net_pnl"] > 0


def test_metrics_complete_and_reconcile():
    rng = np.random.default_rng(1)
    n = 3000
    idx = pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC")
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    open_ = close.shift(1).fillna(close.iloc[0])
    df = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.002,
                       "low": np.minimum(open_, close) * 0.998, "close": close,
                       "is_filled": False})
    signal = (close > close.rolling(24).mean()).astype(float)
    res = run_backtest(df, signal, FEE, CAP)
    m = compute_metrics(res, df)
    assert all(k in m for k in REQUIRED_KEYS)
    assert m["total_closed_trades"] == len(res.trades) > 10
    assert m["gross_profit"] + m["gross_loss"] == pytest.approx(m["net_profit"])
    assert m["net_profit"] == pytest.approx(res.equity.iloc[-1] - CAP)
    assert 0 <= m["max_drawdown_pct"] <= 100 and 0 <= m["win_rate_pct"] <= 100