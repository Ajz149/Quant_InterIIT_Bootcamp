"""Engine sanity runs (train period only): flat, buy&hold, naive EMA cross."""
import pandas as pd

from src.backtest.engine import run_backtest
from src.backtest.metrics import compute_metrics, format_metrics, quarterly_table
from src.backtest.plots import plot_report
from src.data.splits import load_cfg
from src.features import indicators as ind


def main() -> None:
    cfg = load_cfg()
    fee = cfg["costs"]["fee_slippage_per_side"]
    cap = float(cfg["execution"]["initial_capital"])
    short = cfg["execution"]["allow_short"]
    lo, hi = cfg["splits"]["train"]

    for sym in cfg["data"]["symbols"]:
        df = pd.read_parquet(f"data/processed/{sym}_1h_B.parquet").loc[lo:hi]
        c = df["close"]
        signals = {
            "flat": pd.Series(0.0, index=df.index),
            "buy_hold": pd.Series(1.0, index=df.index),
            "ema_cross_24_96": (ind.ema(c, 24) > ind.ema(c, 96)).astype(float),
        }
        print(f"\n######## {sym}  train {df.index.min():%Y-%m-%d} -> {df.index.max():%Y-%m-%d} ########")
        for name, sig in signals.items():
            res = run_backtest(df, sig, fee=fee, initial_capital=cap, allow_short=short)
            m = compute_metrics(res, df)
            print(f"{name:<16} trades={m['total_closed_trades']:>4}  "
                  f"return={m['total_return_pct']:>8.2f}%  "
                  f"buy&hold={m['buy_hold_return_pct']:>8.2f}%  "
                  f"maxDD={m['max_drawdown_pct']:>6.2f}%")
            if name == "ema_cross_24_96":
                print("\n" + format_metrics(m))
                print("\n" + quarterly_table(res, df).round(2).to_string())
                out = f"results/{sym[:3].lower()}/sanity_ema_cross.png"
                plot_report(df, res, f"{sym} EMA(24/96) cross (train, sanity only)", out)
                print(f"\nplot saved: {out}")


if __name__ == "__main__":
    main()