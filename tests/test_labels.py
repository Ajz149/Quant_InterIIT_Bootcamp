import numpy as np
import pandas as pd

from src.data.splits import get_split
from src.features.labels import triple_barrier

H = 10


def flat_frame(n=60):
    idx = pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}, index=idx)


def test_take_profit_hit():
    df = flat_frame()
    df.iloc[30, df.columns.get_loc("high")] = 110.0      # spike up at bar 30
    out = triple_barrier(df, horizon=H, k_up=3, k_dn=3)
    assert out["tb_label"].iloc[25] == 1
    assert out["tb_hold"].iloc[25] == 5                  # bars 26..30


def test_stop_wins_when_both_barriers_hit_same_bar():
    df = flat_frame()
    df.iloc[30, df.columns.get_loc("high")] = 110.0
    df.iloc[30, df.columns.get_loc("low")] = 90.0
    out = triple_barrier(df, horizon=H, k_up=3, k_dn=3)
    assert out["tb_label"].iloc[25] == -1


def test_timeout_and_unlabeled_tail():
    df = flat_frame()
    df.iloc[30, df.columns.get_loc("high")] = 110.0
    out = triple_barrier(df, horizon=H, k_up=3, k_dn=3)
    assert out["tb_label"].iloc[40] == 0 and out["tb_hold"].iloc[40] == H
    assert out["tb_label"].iloc[-(H + 1):].isna().all()  # no future left to label


def test_purge_keeps_train_labels_out_of_validation():
    idx = pd.date_range("2023-11-01", "2024-02-01", freq="1h", tz="UTC")
    df = pd.DataFrame({"tb_label": 1.0}, index=idx)
    cfg = {"splits": {"train": ["2021-01-01", "2023-12-31"],
                      "validation": ["2024-01-01", "2024-12-31"]}}
    purge = H + 1
    train = get_split(df, cfg, "train", purge)
    val_start = pd.Timestamp("2024-01-01", tz="UTC")
    assert train.index.max() + pd.Timedelta(hours=H) < val_start   # label window ends before val