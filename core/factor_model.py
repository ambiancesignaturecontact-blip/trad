"""
Factor model + risk budgeting (VISION §6).

- compute_factor_exposures(): returns exposures of an equity series to the classic
  factors (market, momentum, carry, volatility) for P&L attribution.
- risk_parity_weights(): weights strategies by inverse realized volatility so that
  each strategy contributes a BUDGET OF RISK (vol), not a budget of capital.
"""
import logging
import numpy as np

logger = logging.getLogger("FactorModel")


def compute_factor_exposures(returns: list, market_returns: list,
                             momentum_returns: list, carry_returns: list,
                             vol_returns: list) -> dict:
    """
    Ordinary least squares exposure of `returns` to 4 factors.
    Returns beta coefficients + R2. Pure numpy (no sklearn needed here).
    """
    n = min(len(returns), len(market_returns), len(momentum_returns), len(carry_returns), len(vol_returns))
    if n < 15:
        return {"valid": False, "reason": "insufficient samples"}

    y = np.array(returns[-n:])
    X = np.column_stack([
        np.array(market_returns[-n:]),
        np.array(momentum_returns[-n:]),
        np.array(carry_returns[-n:]),
        np.array(vol_returns[-n:]),
        np.ones(n),
    ])
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ beta
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        return {
            "valid": True,
            "beta_market": round(float(beta[0]), 4),
            "beta_momentum": round(float(beta[1]), 4),
            "beta_carry": round(float(beta[2]), 4),
            "beta_vol": round(float(beta[3]), 4),
            "alpha": round(float(beta[4]), 6),
            "r2": round(float(r2), 4),
            "samples": n,
        }
    except Exception as e:
        return {"valid": False, "reason": str(e)}


def risk_parity_weights(strategy_returns: dict, min_weight: float = 0.02,
                        max_weight: float = 0.35) -> dict:
    """
    VISION §6: budget of RISK per strategy = weight proportional to 1/vol.
    Strategies with high recent volatility get less capital so each contributes
    a comparable amount of risk.
    """
    weights = {}
    for name, rets in strategy_returns.items():
        if len(rets) < 5:
            weights[name] = 1.0  # unknown vol -> neutral
            continue
        vol = float(np.std(rets[-40:])) + 1e-9
        weights[name] = 1.0 / vol
    total = sum(weights.values())
    if total <= 0:
        return {k: 1.0 / max(len(weights), 1) for k in weights}
    out = {k: v / total for k, v in weights.items()}
    # clip to bounds and renormalize
    clipped = {k: float(np.clip(v, min_weight, max_weight)) for k, v in out.items()}
    s = sum(clipped.values())
    return {k: round(v / s, 4) for k, v in clipped.items()}
