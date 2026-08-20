"""
Signal library + statistical admission gate (VISION §2.1, §4.4).

A catalogue of pure signal functions (features -> score in [-1,1]) that can be
evaluated in batch over historical data. Each signal is admitted into live use
ONLY if its Deflated Sharpe Ratio (corrected for data-snooping across the
catalogue) exceeds a threshold - the same discipline used by top quant funds.
"""
import logging
from collections.abc import Callable

import numpy as np
import pandas as pd

from models.lopez_de_prado import calculate_deflated_sharpe_ratio

logger = logging.getLogger("SignalLibrary")


# --------------------------------------------------------------------------
# Pure signals: fn(df, market_data) -> float in [-1, 1]
# --------------------------------------------------------------------------
def sig_momentum_roc(df, md, period=24):
    c = df["close"].values
    if len(c) < period + 5:
        return 0.0
    roc = (c[-1] - c[-period]) / max(c[-period], 1e-9)
    return float(np.clip(np.tanh(roc * 20.0), -1.0, 1.0))


def sig_momentum_cross(df, md, fast=12, slow=26):
    c = df["close"].values
    if len(c) < slow + 5:
        return 0.0
    s = pd.Series(c)
    ema_f = s.ewm(span=fast).mean().iloc[-1]
    ema_s = s.ewm(span=slow).mean().iloc[-1]
    return float(np.clip((ema_f - ema_s) / max(ema_s, 1e-9) * 200.0, -1.0, 1.0))


def sig_rsi_meanrev(df, md, period=14):
    c = df["close"].values
    if len(c) < period + 5:
        return 0.0
    d = pd.Series(c).diff()
    gain = d.where(d > 0, 0).rolling(period).mean().iloc[-1]
    loss = (-d.where(d < 0, 0)).rolling(period).mean().iloc[-1]
    rs = gain / max(loss, 1e-9)
    rsi = 100 - 100 / (1 + rs)
    # contrarian: overbought -> sell
    return float(np.clip((50 - rsi) / 30.0, -1.0, 1.0))


def sig_bollinger_revert(df, md, period=20, std=2.0):
    c = df["close"].values
    if len(c) < period + 5:
        return 0.0
    s = pd.Series(c)
    sma = s.rolling(period).mean().iloc[-1]
    sd = s.rolling(period).std().iloc[-1]
    z = (c[-1] - sma) / max(sd, 1e-9)
    return float(np.clip(-z / std, -1.0, 1.0))


def sig_vol_breakout(df, md, period=20):
    c, h, lo = df["close"].values, df["high"].values, df["low"].values
    if len(c) < period + 5:
        return 0.0
    tr = np.maximum(h[1:] - lo[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(lo[1:] - c[:-1])))
    atr = pd.Series(tr).rolling(period).mean().iloc[-1]
    recent_vol = float(np.std(c[-period:]))
    if recent_vol <= 1e-9:
        return 0.0
    # expansion ratio vs ATR baseline
    ratio = recent_vol / max(atr, 1e-9)
    direction = 1.0 if c[-1] > c[-period] else -1.0
    return float(np.clip(direction * (ratio - 1.0) * 3.0, -1.0, 1.0))


def sig_vpin(df, md):
    vpin = md.get("vpin")
    if vpin is None:
        return 0.0
    # high VPIN = toxic flow -> defensive short bias
    return float(np.clip((0.5 - float(vpin)) * 2.0, -1.0, 1.0))


def sig_carry(df, md):
    funding = md.get("funding_rate_8h")
    if funding is None:
        return 0.0
    cap = 0.0005
    return float(np.clip(float(funding) / cap, -1.0, 1.0))


def sig_cross_sectional(df, md):
    spread = md.get("market_avg_return")
    if spread is None:
        return 0.0
    c = df["close"].values
    if len(c) < 25:
        return 0.0
    roc = (c[-1] - c[-24]) / max(c[-24], 1e-9)
    return float(np.clip((roc - float(spread)) / 0.05, -1.0, 1.0))


def sig_onchain(df, md):
    onchain = md.get("onchain_risk")
    if onchain is None:
        return 0.0
    # low on-chain risk -> accumulation bias (VISION §5: on-chain as alpha)
    return float(np.clip((0.5 - float(onchain)) * 2.0, -1.0, 1.0))


def sig_sentiment(df, md):
    sent = md.get("sentiment")
    if sent is None:
        return 0.0
    return float(np.clip(float(sent), -1.0, 1.0))


def sig_volume_confirmation(df, md, period=20):
    v = df["volume"].values
    if len(v) < period + 2:
        return 0.0
    c = df["close"].values
    vol_ratio = v[-1] / max(np.mean(v[-period:]), 1e-9)
    direction = 1.0 if c[-1] > c[-2] else -1.0
    return float(np.clip(direction * (vol_ratio - 1.0) * 2.0, -1.0, 1.0))


SIGNAL_LIBRARY: dict[str, Callable] = {
    "momentum_roc": sig_momentum_roc,
    "momentum_cross": sig_momentum_cross,
    "rsi_meanrev": sig_rsi_meanrev,
    "bollinger_revert": sig_bollinger_revert,
    "vol_breakout": sig_vol_breakout,
    "vpin_defensive": sig_vpin,
    "carry": sig_carry,
    "cross_sectional": sig_cross_sectional,
    "onchain_alpha": sig_onchain,
    "sentiment": sig_sentiment,
    "volume_confirmation": sig_volume_confirmation,
}


def evaluate_signal(df: pd.DataFrame, signal_fn: Callable,
                    market_data: dict = None, fee_pct: float = 0.001) -> dict:
    """
    Batch evaluation of one signal over the full history (VISION §2.1):
    signal series -> hypothetical PnL (net of round-trip fee) -> Sharpe + DSR.
    """
    md = market_data or {}
    closes = df["close"].values
    if len(closes) < 60:
        return {"valid": False, "reason": "insufficient bars"}

    signals = []
    warmup = 40
    for i in range(warmup, len(df)):
        window = df.iloc[:i + 1]
        md_i = dict(md)
        md_i["df"] = window
        md_i["price_primary"] = float(closes[i])
        md_i["price_secondary"] = float(closes[i])
        signals.append(signal_fn(window, md_i))
    signals = np.array(signals)

    rets = np.diff(closes[warmup - 1:]) / closes[warmup - 1:-1]
    n = min(len(signals) - 1, len(rets))
    if n < 20:
        return {"valid": False, "reason": "insufficient signals"}

    strat_rets = signals[:n] * rets[:n]
    # round-trip fee on flips
    flips = np.sum(np.abs(np.diff(np.sign(signals[:n + 1]))) > 0)
    strat_rets = strat_rets - fee_pct * flips / max(n, 1)

    sharpe = 0.0
    if strat_rets.std() > 0:
        sharpe = float(strat_rets.mean() / strat_rets.std() * np.sqrt(365 * 24))

    dsr = calculate_deflated_sharpe_ratio(
        observed_sharpe=sharpe,
        num_trials=len(SIGNAL_LIBRARY),
        trials_variance_sharpe=0.1,
        sample_length=n,
    )
    return {
        "valid": True,
        "sharpe": round(sharpe, 3),
        "deflated_sharpe": round(dsr, 4),
        "bars": n,
        "flips": int(flips),
    }


def evaluate_all_signals(df: pd.DataFrame, market_data: dict = None) -> dict[str, dict]:
    """Evaluates the whole catalogue and returns ranked results (admission gate)."""
    results = {}
    for name, fn in SIGNAL_LIBRARY.items():
        try:
            results[name] = evaluate_signal(df, fn, market_data)
        except Exception as e:
            results[name] = {"valid": False, "reason": str(e)}
    # rank by deflated sharpe
    ranked = sorted(
        [(k, v) for k, v in results.items() if v.get("valid")],
        key=lambda kv: kv[1].get("deflated_sharpe", 0.0),
        reverse=True,
    )
    return {"results": results, "ranking": [k for k, _ in ranked]}
