"""
VISION_FUTUR §8 - L'INDICE DE CONFIANCE (honnêteté finale).

Composite index that makes the bot distrust itself when it should:
- simulation<->live slippage divergence
- live p-value of recent performance (is it luck?)
- paper<->live gap (via slippage model trust)
- data quality
When the index falls below a threshold, the bot AUTOMATICALLY shrinks sizes.
"""
import logging

import numpy as np

logger = logging.getLogger("ConfidenceIndex")


def compute_confidence_index(sim_divergence: float, p_value: float,
                             data_quality: str, slippage_trust: float = 1.0,
                             base: float = 100.0) -> dict:
    """
    Returns {index (0..100), factor (size multiplier 0.3..1.0), reasons}.
    """
    score = base
    reasons = []

    # 1. sim<->live divergence (up to -30)
    if sim_divergence > 1.0:
        score -= 30
        reasons.append(f"divergence simulé/réel {sim_divergence:.1f}x")
    elif sim_divergence > 0.5:
        score -= 15
        reasons.append("divergence simulé/réel élevée")

    # 2. live p-value: low p = performance real (no penalty); high p = luck
    if p_value > 0.30:
        score -= 25
        reasons.append(f"p-value {p_value:.2f} (performance possiblement due au hasard)")
    elif p_value > 0.15:
        score -= 10

    # 3. data quality
    dq_penalty = {"LIVE": 0, "DELAYED": 10, "STALE": 20, "UNAVAILABLE": 35,
                  "INVALID": 35, "DISCONNECTED": 35}.get(data_quality, 20)
    if dq_penalty:
        score -= dq_penalty
        reasons.append(f"qualité données {data_quality}")

    # 4. slippage model trust
    score -= int((1.0 - float(np.clip(slippage_trust, 0.0, 1.0))) * 10)

    index = int(np.clip(score, 0, 100))
    factor = 1.0 if index >= 70 else (0.3 + 0.7 * (index / 70.0))
    return {"index": index, "factor": round(float(factor), 3), "reasons": reasons}
