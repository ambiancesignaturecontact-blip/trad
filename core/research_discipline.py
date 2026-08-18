"""
VISION_FUTUR §2 - LA DISCIPLINE DE RECHERCHE ABSOLUE.

- pre-registration: hypotheses registered BEFORE testing (no p-hacking)
- double validation: promotion requires Deflated-Sharpe on the TRAIN slice AND
  a confirmation on a held-out TEST slice (walk-forward + paper-style)
- live p-value: rolling probability that current performance is due to chance
- meta-labeling filter: only execute trades whose predicted success prob is high
"""
import logging
import time
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("ResearchDiscipline")


def pre_register_hypothesis(db, name: str, params: dict, metric: str = "deflated_sharpe",
                            threshold: float = 0.95) -> int:
    """§2a: register the hypothesis + metric + threshold BEFORE running the test."""
    if db is None:
        return 0
    try:
        eid = db.add_experiment(
            hypothesis=f"[PREREG] {name} params={params}",
            status="PREREGISTERED",
            result=f"metric={metric} threshold={threshold}",
        )
        return eid
    except Exception as e:
        logger.warning(f"pre-register failed: {e}")
        return 0


def double_validation(evaluate_fn, df: object, market_data: dict,
                      train_ratio: float = 0.6, promotion_threshold: float = 0.95) -> dict:
    """
    §2b: run the evaluation on a TRAIN slice (for the gate) then CONFIRM on the
    held-out TEST slice. Promotion requires BOTH to pass.
    Returns {train_dsr, test_dsr, promoted}.
    """
    n = len(df)
    split = int(n * train_ratio)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]
    try:
        r_train = evaluate_fn(train_df, market_data)
        r_test = evaluate_fn(test_df, market_data)
        train_dsr = float(r_train.get("deflated_sharpe", 0.0)) if r_train.get("valid") else 0.0
        test_dsr = float(r_test.get("deflated_sharpe", 0.0)) if r_test.get("valid") else 0.0
        promoted = train_dsr >= promotion_threshold and test_dsr >= promotion_threshold
        return {"train_dsr": round(train_dsr, 4), "test_dsr": round(test_dsr, 4),
                "promoted": promoted, "train": r_train, "test": r_test}
    except Exception as e:
        return {"train_dsr": 0.0, "test_dsr": 0.0, "promoted": False, "error": str(e)}


def live_p_value(signals: List[float], returns: List[float]) -> float:
    """
    §2c: rolling probability the current directional accuracy is due to chance
    (binomial test). Low p-value = performance is NOT luck.
    """
    n = min(len(signals), len(returns))
    if n < 20:
        return 0.5  # unknown -> neutral
    correct = sum(1 for i in range(-n, 0) if np.sign(signals[i]) == np.sign(returns[i]))
    p = correct / n
    # binomial two-sided approximation (normal)
    se = np.sqrt(0.25 / n)
    z = (p - 0.5) / max(se, 1e-9)
    from scipy.stats import norm
    p_value = float(2.0 * (1.0 - norm.cdf(abs(z))))
    return round(min(p_value, 1.0), 4)


def meta_label_filter(strategy: str, win_rates: Dict[str, float],
                      threshold: float = 0.52,
                      counts: Optional[Dict[str, int]] = None,
                      min_samples: int = 0) -> bool:
    """
    §2d: meta-labeling - only trade when the strategy's recent win rate exceeds
    the threshold. This is the López de Prado spirit deployed with real outcomes.

    LOT 2 (PDF Pilier D/F) : warm-up anti-verrouillage — tant qu'une stratégie
    a moins de `min_samples` trades CLÔTURÉS réels, on laisse passer (sinon le
    bot ne pourrait jamais constituer d'historique). Comportement par défaut
    inchangé (min_samples=0) pour préserver la compatibilité.
    """
    wr = float(win_rates.get(strategy, 0.0))
    if counts is not None and min_samples > 0:
        if int(counts.get(strategy, 0)) < min_samples:
            return True  # warm-up : pas encore assez d'échantillon réel
    return wr >= threshold
