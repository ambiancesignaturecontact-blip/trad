"""
Volatility targeting overlay (VISION §4.1, §5) - the institutional standard.

Exposure is scaled so that the PORTFOLIO's realized volatility stays near a
target instead of a fixed notional. Scale = target_vol / realized_vol, clipped.
Works on per-tick vol so it is independent of the loop frequency.
"""
import logging

import numpy as np

from core.config import settings

logger = logging.getLogger("VolTargeting")

# P1-8 (audit §3) : défauts branchés sur core/config.py (config.yaml).
_TARGET_TICK_VOL = settings.get_float("vol_targeting", "target_tick_vol", 0.0004)
_REALIZED_WINDOW = settings.get_int("vol_targeting", "realized_window", 40)
_MIN_SCALE = settings.get_float("vol_targeting", "min_scale", 0.25)
_MAX_SCALE = settings.get_float("vol_targeting", "max_scale", 2.0)


def volatility_scale_factor(equity_history: list,
                            target_tick_vol: float = _TARGET_TICK_VOL,
                            realized_window: int = _REALIZED_WINDOW,
                            min_scale: float = _MIN_SCALE,
                            max_scale: float = _MAX_SCALE) -> float:
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
