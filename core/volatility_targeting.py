"""
Volatility targeting overlay (VISION §4.1, §5) - the institutional standard.

Exposure is scaled so that the PORTFOLIO's realized volatility stays near a
target instead of a fixed notional. Scale = target_vol / realized_vol, clipped.
Works on per-tick vol so it is independent of the loop frequency.
"""
import logging
import numpy as np

logger = logging.getLogger("VolTargeting")


def volatility_scale_factor(equity_history: list,
                            target_tick_vol: float = 0.0004,
                            realized_window: int = 40,
                            min_scale: float = 0.25,
                            max_scale: float = 2.0) -> float:
    """
    Returns the multiplier to apply to position sizes.
    target_tick_vol is the desired std of equity returns per loop iteration
    (default 0.0004 ~ 0.04% per tick). Clipped to [min_scale, max_scale].
    Neutral (1.0) when there is not enough history.
    """
    if not equity_history:
        return 1.0
    eq = np.array([float(e) for e in equity_history if isinstance(e, (int, float)) and e > 0])
    if len(eq) < 12:
        return 1.0
    rets = np.diff(eq) / eq[:-1]
    rets = rets[-realized_window:]
    realized_vol = float(np.std(rets))
    if realized_vol <= 1e-9:
        return 1.0
    scale = float(np.clip(target_tick_vol / realized_vol, min_scale, max_scale))
    return scale
