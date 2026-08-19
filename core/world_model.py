"""
VISION §1 - PENSER: a model of the world instead of signal->weights.

- soft regime probabilities (HMM forward probabilities)
- joint market state (vol/liquidity/correlation/macro/sentiment/on-chain)
- causal graph (PC-lite on real features -> returns) so only causal parents trade
- counterfactual marginal alpha per closed trade
"""
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("WorldModel")

FEATURE_NAMES = ["momentum", "vpin", "kyle", "sentiment", "onchain", "funding", "volume"]


def compute_regime_probs(regime_detector, features: np.ndarray) -> Dict[str, float]:
    """Soft probabilities P(regime | data) via the HMM forward pass."""
    try:
        if features.ndim == 1:
            features = features.reshape(1, -1)
        probs = regime_detector.predict_proba(features)
        last = probs[-1]
        return {str(i): round(float(p), 4) for i, p in enumerate(last)}
    except Exception as e:
        logger.warning(f"regime probs failed: {e}")
        return {}


def compute_market_state(state: dict, regime_probs: dict, vol_mean: float = 0.0,
                         avg_correlation: float = 0.0) -> dict:
    """Joint market state - the 'N-dimensional weather' that conditions decisions."""
    try:
        vols = [a.get("price") for a in state.get("assets", {}).values()
                if isinstance(a.get("price"), (int, float))]
        vol_regime = "high" if vol_mean > 0.004 else ("low" if vol_mean < 0.001 else "normal")
        corr_regime = "high" if avg_correlation > 0.7 else ("low" if avg_correlation < 0.3 else "normal")
        return {
            "regime_probs": regime_probs,
            "vol_mean": round(float(vol_mean), 6),
            "vol_regime": vol_regime,
            "correlation": round(float(avg_correlation), 3),
            "corr_regime": corr_regime,
            # FIX (logs) : sentiment/onchain peuvent être None (source
            # indisponible) -> neutre 0.0, jamais de float(None) qui spamme
            "sentiment": float(state.get("sentiment_index") or 0.0),
            "onchain_risk": float(state.get("onchain_risk_score") or 0.0),
            "data_quality": state.get("data_quality_status", "UNAVAILABLE"),
            "n_assets": len(vols),
        }
    except Exception as e:
        logger.warning(f"market state failed: {e}")
        return {}


# ---------------------------------------------------------------------------
# Causal graph (PC-lite): partial-correlation based parent discovery on REAL data
# ---------------------------------------------------------------------------
def _partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Partial correlation of x,y given z (numpy OLS residuals)."""
    def resid(a, b):
        if b.ndim == 1:
            b = b.reshape(-1, 1)
        b = np.column_stack([b, np.ones(len(b))])
        try:
            beta, *_ = np.linalg.lstsq(b, a, rcond=None)
            return a - b @ beta
        except Exception:
            return a - np.mean(a)
    rx = resid(x, z)
    ry = resid(y, z)
    denom = np.sqrt(np.sum(rx ** 2) * np.sum(ry ** 2)) + 1e-12
    return float(np.sum(rx * ry) / denom)


def discover_causal_parents(features_df: pd.DataFrame, target: str = "returns",
                            alpha: float = 0.05, min_samples: int = 40) -> List[str]:
    """
    PC-lite: returns the features that are partial-correlated with the target
    given the other features (a causal parent set). Real-data only.
    """
    cols = [c for c in features_df.columns if c != target]
    if len(features_df) < min_samples or not cols:
        return []
    y = features_df[target].values
    parents = []
    for c in cols:
        x = features_df[c].values
        others = np.column_stack([features_df[o].values for o in cols if o != c])
        pc = abs(_partial_corr(x, y, others))
        # crude significance: |partial corr| > 2/sqrt(n)
        if pc > 2.0 / np.sqrt(len(features_df)):
            parents.append(c)
    return parents


def build_causal_feature_df(state: dict, df: pd.DataFrame, market_data: dict) -> pd.DataFrame:
    """Builds a REAL feature matrix from the live market_data for causal analysis."""
    close = df["close"].values
    rets = np.diff(close) / np.maximum(close[:-1], 1e-9)
    n = len(rets)
    if n < 10:
        return pd.DataFrame()
    data = {
        "returns": rets[-n:],
        "momentum": np.gradient(close[-n:]) / np.maximum(close[-n:], 1e-9),
        "vpin": [float(market_data.get("vpin", 0.5))] * n,
        "kyle": [float(market_data.get("kyle_lambda", 0.0))] * n,
        "sentiment": [float(state.get("sentiment_index", 0.0))] * n,
        "onchain": [float(state.get("onchain_risk_score", 0.5))] * n,
        "funding": [float(state.get("funding_rates", {}).get(market_data.get("symbol", ""), 0.0))] * n,
        "volume": np.gradient(df["volume"].values[-n:]),
    }
    return pd.DataFrame(data)


def counterfactual_alpha(trade: dict, benchmark_return: float) -> float:
    """
    VISION §1d: marginal alpha of a closed trade vs the benchmark.
    trade: {side, qty, entry, exit}
    Returns the excess return the trade captured (positive = good decision).
    """
    try:
        entry, exit_px = float(trade["entry"]), float(trade["exit"])
        side = str(trade.get("side", "BUY")).upper()
        ret = (exit_px - entry) / max(entry, 1e-9)
        if side == "SELL":
            ret = -ret
        return float(ret - benchmark_return)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# VISION_FUTUR §3: structural regimes + cross-asset context
# ---------------------------------------------------------------------------
def compute_structural_regimes(state: dict, spread_bps: float = 2.0,
                               avg_correlation: float = 0.0) -> dict:
    """
    Structural market regimes beyond volatility:
    liquidity (spread), sentiment, on-chain, correlation, macro.
    """
    try:
        # FIX (logs) : mêmes gardes None -> neutre
        sentiment = float(state.get("sentiment_index") or 0.0)
        onchain = float(state.get("onchain_risk_score") or 0.5)
        liq_regime = "tight" if spread_bps < 3 else ("wide" if spread_bps > 15 else "normal")
        sent_regime = "bullish" if sentiment > 0.2 else ("bearish" if sentiment < -0.2 else "neutral")
        onchain_regime = "risky" if onchain > 0.75 else ("healthy" if onchain < 0.4 else "normal")
        return {
            "liquidity": liq_regime,
            "sentiment": sent_regime,
            "onchain": onchain_regime,
            "correlation": round(float(avg_correlation), 3),
            "spread_bps": round(float(spread_bps), 2),
        }
    except Exception as e:
        logger.warning(f"structural regimes failed: {e}")
        return {}


def cross_asset_bias(current_symbol: str, state: dict) -> float:
    """
    VISION_FUTUR §3/§4: cross-asset learning - the BTC regime informs other
    assets. Returns a bias in [-0.3, +0.3] applied to non-BTC signals.
    """
    try:
        probs = state.get("regime_probs", {}) or {}
        p_bull = float(probs.get("0", 0.0))
        p_bear = float(probs.get("1", 0.0))
        net = p_bull - p_bear
        if current_symbol == "BTCUSDT":
            return 0.0
        # in a strong bull/bear, correlated assets lean the same way (soft)
        return float(np.clip(net * 0.3, -0.3, 0.3))
    except Exception:
        return 0.0
