"""
EDGE DECAY ENGINE (LOT 4 du mandat — Adaptativité).

Détecte automatiquement qu'une stratégie perd son edge et y répond de façon
BORNÉE et RÉVERSIBLE. Cycle d'états :

    HEALTHY -> DEGRADED -> WARNING -> DISABLED -> RECOVERY

Surveillé par stratégie (et par régime) :
  - expectancy (EMA des pnl% par trade clôturé)
  - hit rate (EMA des issues gagnantes)
  - profit factor (somme des gains / |somme des pertes|)
  - coût moyen par trade (quand fourni)
  - performance par régime (pnl cumulé par régime HMM)

Principes (mandat : « une stratégie dégradée ne doit pas être supprimée ») :
  1. JAMAIS de suppression dure : le scale de poids est borné [0.30, 1.0].
     DISABLED = sous-pondération à 0.30 (observation), pas un arrêt.
  2. RÉVERSIBLE : une stratégie en RECOVERY (expectancy redevenue positive
     sur échantillon) retrouve progressivement son poids.
  3. Pas de jugement sur échantillon trop faible : n < MIN_SAMPLES -> HEALTHY
     (on n'invente pas une dégradation sur 3 trades).
  4. La taille finale reste gouvernée par le Risk Engine : ce module ne fait
     que recommander un scale de pondération à la meta-allocation.
  5. DÉMO == RÉAL : aucun flag de mode.

Branchement :
  - main.py (record_closed_trade) : edge_decay.record_outcome(strategy, pnl_pct, regime)
  - strategies/engine.py (allocate) : edge_decay_scales={strategy: scale}
    multiplie le poids final de chaque stratégie.
"""
import logging
import time
from collections import deque

from core.config import settings

logger = logging.getLogger("InstitutionalTradingBot")  # même canal que main

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
EDGE_MIN_SAMPLES: int = settings.get_int("edge_decay", "min_samples", 10)
EDGE_EMA_ALPHA: float = settings.get_float("edge_decay", "ema_alpha", 0.20)
EDGE_WARN_EXPECTANCY: float = settings.get_float("edge_decay", "warn_expectancy", -0.001)
EDGE_DISABLE_EXPECTANCY: float = settings.get_float("edge_decay", "disable_expectancy", -0.003)
EDGE_RECOVER_EXPECTANCY: float = settings.get_float("edge_decay", "recover_expectancy", 0.0005)
EDGE_SCALE_DISABLED: float = settings.get_float("edge_decay", "scale_disabled", 0.30)
EDGE_SCALE_WARNING: float = settings.get_float("edge_decay", "scale_warning", 0.60)
EDGE_SCALE_DEGRADED: float = settings.get_float("edge_decay", "scale_degraded", 0.85)
EDGE_SCALE_RECOVERY: float = settings.get_float("edge_decay", "scale_recovery", 0.85)
EDGE_HISTORY_MAXLEN: int = settings.get_int("edge_decay", "history_maxlen", 200)

# États (cycle documenté)
HEALTHY = "HEALTHY"
DEGRADED = "DEGRADED"
WARNING = "WARNING"
DISABLED = "DISABLED"
RECOVERY = "RECOVERY"


def _ema(previous, value, alpha: float = EDGE_EMA_ALPHA):
    return alpha * value + (1.0 - alpha) * previous


class StrategyEdge:
    """État de l'edge d'UNE stratégie."""

    def __init__(self, name: str):
        self.name = name
        self.pnls: deque = deque(maxlen=EDGE_HISTORY_MAXLEN)
        self.expectancy: float | None = None      # EMA des pnl%
        self.hit_rate: float | None = None        # EMA des issues
        self.regime_pnl: dict[int, list[float]] = {}  # régime -> pnls
        self.state: str = HEALTHY
        self.prev_state: str = HEALTHY
        self.updated_ts: float = 0.0
        self.cost_pnl: deque = deque(maxlen=EDGE_HISTORY_MAXLEN)  # pnl nets de coûts quand fournis

    @property
    def n(self) -> int:
        return len(self.pnls)

    def record(self, pnl_pct: float, regime_id: int | None = None,
               pnl_net_cost: float | None = None) -> None:
        self.pnls.append(float(pnl_pct))
        if pnl_net_cost is not None:
            self.cost_pnl.append(float(pnl_net_cost))
        # expectancy EMA (a priori neutre 0.0 au premier trade)
        prev_exp = self.expectancy if self.expectancy is not None else 0.0
        self.expectancy = _ema(prev_exp, float(pnl_pct))
        # hit rate EMA (a priori neutre 0.5)
        prev_hr = self.hit_rate if self.hit_rate is not None else 0.5
        self.hit_rate = _ema(prev_hr, 1.0 if pnl_pct > 0 else 0.0)
        if regime_id is not None:
            self.regime_pnl.setdefault(int(regime_id), []).append(float(pnl_pct))
            rl = self.regime_pnl[int(regime_id)]
            if len(rl) > EDGE_HISTORY_MAXLEN:
                del rl[: len(rl) - EDGE_HISTORY_MAXLEN]
        self.updated_ts = time.time()
        self._refresh_state()

    def _refresh_state(self) -> None:
        """Met à jour l'état selon l'expectancy EMA et l'échantillon."""
        if self.expectancy is None or self.n < EDGE_MIN_SAMPLES:
            self.state = HEALTHY  # pas de jugement sur échantillon faible
            return
        exp = self.expectancy
        if exp <= EDGE_DISABLE_EXPECTANCY:
            new = DISABLED
        elif exp <= EDGE_WARN_EXPECTANCY:
            new = WARNING
        elif exp >= EDGE_RECOVER_EXPECTANCY:
            # RECOVERY si la stratégie SORT d'un état dégradé (l'expectancy
            # remonte en passant par DEGRADED/WARNING — prev_state vaut alors
            # DEGRADED, pas DISABLED). Une stratégie restée saine reste HEALTHY.
            new = RECOVERY if self.prev_state != HEALTHY else HEALTHY
        else:
            new = DEGRADED
        self.prev_state = self.state
        self.state = new

    def weight_scale(self) -> float:
        """Scale de pondération borné [0.30, 1.0] — JAMAIS 0 (pas de
        suppression dure ; DISABLED = observation à 30 %)."""
        return {
            HEALTHY: 1.0,
            DEGRADED: EDGE_SCALE_DEGRADED,
            WARNING: EDGE_SCALE_WARNING,
            DISABLED: EDGE_SCALE_DISABLED,
            RECOVERY: EDGE_SCALE_RECOVERY,
        }.get(self.state, 1.0)

    def profit_factor(self) -> float | None:
        if self.n < 2:
            return None
        wins = sum(p for p in self.pnls if p > 0)
        losses = abs(sum(p for p in self.pnls if p < 0))
        if losses < 1e-12:
            return None
        return wins / losses

    def avg_cost_pct(self) -> float | None:
        if not self.cost_pnl:
            return None
        # coût moyen ≈ |pnl brut - pnl net| par trade (si net fourni)
        costs = [abs(a - b) for a, b in zip(self.pnls, self.cost_pnl)
                 if b is not None]
        if not costs:
            return None
        return sum(costs) / len(costs)

    def regime_summary(self) -> dict:
        out = {}
        for rid, pnls in self.regime_pnl.items():
            out[str(rid)] = {
                "n": len(pnls),
                "pnl_sum_pct": round(sum(pnls) * 100.0, 4),
                "avg_pct": round((sum(pnls) / len(pnls)) * 100.0, 4),
            }
        return out

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "n": self.n,
            "expectancy_ema_pct": round(self.expectancy * 100.0, 4) if self.expectancy is not None else None,
            "hit_rate_ema": round(self.hit_rate, 4) if self.hit_rate is not None else None,
            "profit_factor": round(self.profit_factor(), 4) if self.profit_factor() is not None else None,
            "avg_cost_pct": round(self.avg_cost_pct() * 100.0, 4) if self.avg_cost_pct() is not None else None,
            "per_regime": self.regime_summary(),
            "weight_scale": self.weight_scale(),
            "updated_ts": round(self.updated_ts, 2),
        }


class EdgeDecayEngine:
    """
    Moteur global : surveille TOUTES les stratégies, recommande les scales
    de pondération, expose le rapport (télémétrie/API).
    """

    def __init__(self, strategies: list[str] | None = None):
        self.edges: dict[str, StrategyEdge] = {}
        if strategies:
            for s in strategies:
                self.edges[s] = StrategyEdge(s)

    def ensure(self, strategy: str) -> StrategyEdge:
        if strategy not in self.edges:
            self.edges[strategy] = StrategyEdge(strategy)
        return self.edges[strategy]

    def record_outcome(self, strategy: str, pnl_pct: float,
                       regime_id: int | None = None,
                       pnl_net_cost: float | None = None) -> None:
        """Enregistre un trade clôturé. JAMAIS bloquant."""
        try:
            if not strategy:
                return
            self.ensure(strategy).record(pnl_pct, regime_id, pnl_net_cost)
        except Exception as e:
            logger.warning(f"edge_decay record_outcome failed ({e})")

    def weight_scale(self, strategy: str) -> float:
        return self.ensure(strategy).weight_scale()

    def scales(self) -> dict[str, float]:
        """{strategy: scale} pour la meta-allocation (borné [0.30, 1.0])."""
        return {s: e.weight_scale() for s, e in self.edges.items()}

    def states(self) -> dict[str, str]:
        return {s: e.state for s, e in self.edges.items()}

    def report(self) -> dict:
        """Rapport complet pour la télémétrie / l'API."""
        per = {s: e.to_dict() for s, e in self.edges.items()}
        n_disabled = sum(1 for e in self.edges.values() if e.state == DISABLED)
        n_warning = sum(1 for e in self.edges.values() if e.state == WARNING)
        return {
            "per_strategy": per,
            "counts": {
                "total": len(self.edges),
                "healthy": sum(1 for e in self.edges.values() if e.state == HEALTHY),
                "degraded": sum(1 for e in self.edges.values() if e.state == DEGRADED),
                "warning": n_warning,
                "disabled": n_disabled,
                "recovery": sum(1 for e in self.edges.values() if e.state == RECOVERY),
            },
            "note": "Une stratégie DISABLED est sous-pondérée (x0.30, jamais arrêtée) "
                    "et peut revenir en RECOVERY si son expectancy redevient positive.",
            "ts": time.time(),
        }
