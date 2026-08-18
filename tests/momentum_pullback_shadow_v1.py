import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import momentum_pullback_shadow as mp


def _assert_close(actual, expected, tolerance=1e-9):
    assert abs(float(actual) - float(expected)) <= tolerance, (actual, expected)


def _base_indicator_frame(length=230):
    index = pd.date_range("2025-01-02", periods=length, freq="B")
    frame = pd.DataFrame(
        {
            "Open": np.full(length, 100.0),
            "High": np.full(length, 101.0),
            "Low": np.full(length, 99.0),
            "Close": np.full(length, 100.0),
            "Volume": np.full(length, 100.0),
            "SMA200": np.full(length, 95.0),
            "EMA20": np.full(length, 100.0),
            "EMA9": np.full(length, 100.0),
            "ATR14": np.full(length, 4.0),
            "RSI14": np.full(length, 50.0),
            "PRIOR_AVG_VOLUME20": np.full(length, 100.0),
            "STOCK_6M_RETURN": np.full(length, 0.1),
            "SPY_6M_RETURN": np.full(length, 0.05),
            "RELATIVE_STRENGTH_EXCESS": np.full(length, 0.05),
        },
        index=index,
    )
    return frame


def _long_confirmed_frame():
    frame = _base_indicator_frame()
    idx = 220
    touch_idx = 218
    frame.iloc[idx - 9 : idx + 1, frame.columns.get_loc("Low")] = 101.0
    frame.iloc[touch_idx, frame.columns.get_loc("Low")] = 99.0
    frame.iloc[touch_idx, frame.columns.get_loc("EMA20")] = 100.0
    frame.iloc[touch_idx, frame.columns.get_loc("ATR14")] = 4.0
    frame.iloc[idx - 1, frame.columns.get_loc("High")] = 101.0
    frame.iloc[idx - 1, frame.columns.get_loc("RSI14")] = 40.0
    frame.iloc[idx, frame.columns.get_loc("Open")] = 100.0
    frame.iloc[idx, frame.columns.get_loc("Close")] = 103.0
    frame.iloc[idx, frame.columns.get_loc("High")] = 104.0
    frame.iloc[idx, frame.columns.get_loc("Low")] = 101.0
    frame.iloc[idx, frame.columns.get_loc("RSI14")] = 44.0
    frame.iloc[idx, frame.columns.get_loc("Volume")] = 150.0
    frame.iloc[idx, frame.columns.get_loc("PRIOR_AVG_VOLUME20")] = 100.0
    return frame, idx


def _short_confirmed_frame():
    frame = _base_indicator_frame()
    idx = 220
    touch_idx = 218
    frame["SMA200"] = 105.0
    frame.iloc[idx - 9 : idx + 1, frame.columns.get_loc("High")] = 99.0
    frame.iloc[touch_idx, frame.columns.get_loc("High")] = 101.0
    frame.iloc[touch_idx, frame.columns.get_loc("EMA20")] = 100.0
    frame.iloc[touch_idx, frame.columns.get_loc("ATR14")] = 4.0
    frame.iloc[idx - 1, frame.columns.get_loc("Low")] = 99.0
    frame.iloc[idx - 1, frame.columns.get_loc("RSI14")] = 62.0
    frame.iloc[idx, frame.columns.get_loc("Open")] = 100.0
    frame.iloc[idx, frame.columns.get_loc("Close")] = 97.0
    frame.iloc[idx, frame.columns.get_loc("High")] = 99.0
    frame.iloc[idx, frame.columns.get_loc("Low")] = 96.0
    frame.iloc[idx, frame.columns.get_loc("RSI14")] = 58.0
    frame.iloc[idx, frame.columns.get_loc("Volume")] = 150.0
    frame.iloc[idx, frame.columns.get_loc("PRIOR_AVG_VOLUME20")] = 100.0
    return frame, idx


def test_sma200():
    values = pd.Series(range(1, 202), dtype=float)
    result = mp.sma(values, 200)
    assert math.isnan(result.iloc[198])
    _assert_close(result.iloc[199], sum(range(1, 201)) / 200.0)
    _assert_close(result.iloc[200], sum(range(2, 202)) / 200.0)


def test_ema20_interaction_and_atr_penetration_rule():
    frame = _base_indicator_frame()
    row = frame.iloc[50].copy()
    row["Low"] = 98.1
    row["EMA20"] = 100.0
    row["ATR14"] = 4.0
    assert mp._ema_interaction_at(row, mp.LONG) is not None
    row["Low"] = 97.9
    assert mp._ema_interaction_at(row, mp.LONG) is None

    row["High"] = 101.9
    row["Low"] = 99.0
    assert mp._ema_interaction_at(row, mp.SHORT) is not None
    row["High"] = 102.1
    assert mp._ema_interaction_at(row, mp.SHORT) is None


def test_rsi_reset_turn_long_and_short():
    long_frame, idx = _long_confirmed_frame()
    ok, details = mp._rsi_reset_turn(long_frame, idx, mp.LONG)
    assert ok
    assert details["entered_reset_range"]
    assert details["rsi_turned"]

    short_frame, idx = _short_confirmed_frame()
    ok, details = mp._rsi_reset_turn(short_frame, idx, mp.SHORT)
    assert ok
    assert details["entered_reset_range"]
    assert details["rsi_turned"]


def test_volume_average_excludes_signal_bar():
    volume = pd.Series([100.0] * 20 + [1000.0])
    prior = mp.prior_average_volume(volume, 20)
    _assert_close(prior.iloc[20], 100.0)


def test_bullish_and_bearish_trigger_conditions():
    long_frame, idx = _long_confirmed_frame()
    ok, details = mp._trigger_conditions(long_frame, idx, mp.LONG)
    assert ok
    assert details["trigger_candle"]
    assert details["trigger_break"]
    assert details["trigger_volume"]

    short_frame, idx = _short_confirmed_frame()
    ok, details = mp._trigger_conditions(short_frame, idx, mp.SHORT)
    assert ok
    assert details["trigger_candle"]
    assert details["trigger_break"]
    assert details["trigger_volume"]


def test_execution_confirmed_long_short_symmetry():
    long_frame, idx = _long_confirmed_frame()
    state, details = mp.evaluate_indicator_frame_at("TEST", long_frame, idx, mp.LONG)
    assert state == mp.STATE_EXECUTION_CONFIRMED
    assert details["ema_interaction"]["bars_since"] == 2

    short_frame, idx = _short_confirmed_frame()
    state, details = mp.evaluate_indicator_frame_at("TEST", short_frame, idx, mp.SHORT)
    assert state == mp.STATE_EXECUTION_CONFIRMED
    assert details["ema_interaction"]["bars_since"] == 2


def test_10_bar_pullback_expiration():
    frame = _base_indicator_frame()
    idx = 220
    frame["Low"] = 101.0
    frame.iloc[idx - 10, frame.columns.get_loc("Low")] = 99.0
    state, details = mp.evaluate_indicator_frame_at("TEST", frame, idx, mp.LONG)
    assert state == mp.STATE_EXPIRED
    assert details["bars_since_ema_interaction"] == 10


def test_next_session_open_entry_and_no_same_bar_entry():
    frame, idx = _long_confirmed_frame()
    state, details = mp.evaluate_indicator_frame_at("TEST", frame, idx, mp.LONG)
    assert state == mp.STATE_EXECUTION_CONFIRMED
    frame.iloc[idx + 1, frame.columns.get_loc("Open")] = 111.0
    signal = mp._signal_from_evaluation("TEST", frame, idx, mp.LONG, details)
    assert signal.signal_timestamp == frame.index[idx].isoformat()
    assert signal.entry_timestamp == frame.index[idx + 1].isoformat()
    assert signal.entry_timestamp != signal.signal_timestamp
    _assert_close(signal.entry_price, 111.0)


def test_no_lookahead_historical_slicing():
    frame = _base_indicator_frame()
    as_of = frame.index[210]
    future = frame.copy()
    future.iloc[211:, future.columns.get_loc("Close")] = 10000.0
    sliced = mp.historical_slice(future, as_of)
    assert sliced.index.max() == as_of
    assert float(sliced["Close"].max()) < 10000.0


def test_append_future_stability():
    frame, idx = _long_confirmed_frame()
    state, details = mp.evaluate_indicator_frame_at("TEST", frame, idx, mp.LONG)
    signal = mp._signal_from_evaluation("TEST", frame, idx, mp.LONG, details)

    future = frame.copy()
    extra_idx = pd.date_range(frame.index[-1] + pd.offsets.BDay(1), periods=20, freq="B")
    extra = frame.iloc[-20:].copy()
    extra.index = extra_idx
    extra["Close"] = 10000.0
    appended = pd.concat([future, extra])
    sliced = appended.loc[appended.index <= frame.index[idx + 1]]
    state2, details2 = mp.evaluate_indicator_frame_at("TEST", sliced, idx, mp.LONG)
    signal2 = mp._signal_from_evaluation("TEST", sliced, idx, mp.LONG, details2)
    assert state2 == state
    assert signal2.setup_id == signal.setup_id
    assert signal2.pullback_generation_id == signal.pullback_generation_id
    assert signal2.signal_generation_id == signal.signal_generation_id


def test_setup_identity_stability():
    ids1 = mp.setup_identity("abc", mp.LONG, "2026-01-05", "2026-01-08")
    ids2 = mp.setup_identity("ABC", mp.LONG, pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-08"))
    assert ids1 == ids2
    assert mp.VERSION in ids1["setup_id"]


def test_duplicate_suppression():
    base = {
        "setup_id": "A",
        "pullback_generation_id": "ABC|LONG|2026-01-05|momentum-pullback-shadow-v1.0",
        "signal_timestamp": "2026-01-08T00:00:00",
    }
    repeat = dict(base, setup_id="B", signal_timestamp="2026-01-09T00:00:00")
    fresh = {
        "setup_id": "C",
        "pullback_generation_id": "ABC|LONG|2026-02-05|momentum-pullback-shadow-v1.0",
        "signal_timestamp": "2026-02-08T00:00:00",
    }
    independent, duplicates = mp.dedupe_independent_signals([base, repeat, fresh])
    assert [row["setup_id"] for row in independent] == ["A", "C"]
    assert [row["setup_id"] for row in duplicates] == ["B"]


def run_all():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")


if __name__ == "__main__":
    run_all()
