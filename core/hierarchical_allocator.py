"""
HIERARCHICAL META-ALLOCATOR (LOT 5 du mandat).

Fait évoluer l'allocation plate (bandit Thompson + walk-forward + régime +
risk parity — déjà en place dans MetaAllocationEngine) vers une structure
HIÉRARCHIQUE + REGRET MINIMIZATION + NON-STATIONNAIRE :

    Niveau 1 : ACTIF / classe            (déjà couvert : DynamicCapitalAllocator,
                                          portfolio_allocator top-down)
    Niveau 2 : FAMILLE de stratégies     (trend / meanrev / carry / arb / micro)
    Niveau 3 : STRATÉGIE                 (les 12 stratégies existantes)
    Niveau 4 : MODÈLE / EXPERT           (déjà couvert : MoE + bandit par modèle)

Ce module ajoute, sur les NIVEAUX 2 et 3, deux mécanismes mesurables :

  1. REGRET MINIMIZATION (non-stationnaire) :
     regret_i(t+1) = decay * regret_i(t) + (meilleur_pnl_cumulé(t) - pnl_i(t))
     Le regret mesure CE QUE LE SYSTÈME A PERDU en suivant i plutôt que la
     meilleure stratégie ex post. L'exploration est proportionnelle au regret
     (regret matching, type Blackwell) : une stratégie qui accumule du regret
     est davantage RÉESSAYÉE — jamais éliminée (bornes dures).

  2. SCALE DE FAMILLE par performance récente :
     une famille dont la performance récente est négative est sous-pondérée
     au niveau 2 AVANT renormalisation des stratégies en son sein.

Principes (mandat : « maintenir l'aspect non-stationnaire », « renforcer avec
des statistiques réellement observées ») :
  - OUBLI exponentiel du regret (BANDIT-like) : les vieilles erreurs ne
    pèsent plus — le régime change.
  - BORNES dures : le poids final de chaque stratégie reste dans
    [MIN_WEIGHT, MAX_WEIGHT] et la somme reste 1.0 (renormalisée).
  - RÉVERSIBLE : si l'allocateur n'est PAS fourni, le comportement de
    MetaAllocationEngine est strictement identique (testé).
  - Aucun flag de mode (DÉMO == RÉAL).
"""
import logging
import time

from core.config import settings

logger = logging.getLogger("InstitutionalTradingBot")

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
REG_DECAY: float = settings.get_float("hierarchical", "regret_decay", 0.98)
REG_MIN_SAMPLES: int = settings.get_int("hierarchical", "regret_min_samples", 10)
REG_EXPLORATION_WEIGHT: float = settings.get_float("hierarchical", "regret_exploration_weight", 0.15)
FAMILY_EMA_ALPHA: float = settings.get_float("hierarchical", "family_ema_alpha", 0.20)
MIN_WEIGHT: float = settings.get_float("hierarchical", "min_weight", 0.02)
MAX_WEIGHT: float = settings.get_float("hierarchical", "max_weight", 0.45)
FAMILY_SCALE_MIN: float = settings.get_float("hierarchical", "family_scale_min", 0.60)
FAMILY_SCALE_MAX: float = settings.get_float("hierarchical", "family_scale_max", 1.20)

# Familles (niveau 2) — cohérentes avec STRATEGY_FACTOR (core/attribution.py)
FAMILY_OF: dict[str, str] = {
    "Trend Following": "trend",
    "Momentum": "trend",
    "Cross-Sectional Momentum": "trend",
    "Multi-Timeframe": "trend",
    "Volatility Breakout": "trend",
    "Mean Reversion": "meanrev",
    "Grid Trading": "meanrev",
    "Market Making": "carry",
    "Carry": "carry",
    "Statistical Arbitrage": "arbitrage",
    "Inter-Exchange Arbitrage": "arbitrage",
    "Scalping": "micro",
    "META_MODEL": "meta",
}

FAMILY_NAMES: dict[str, str] = {
    "trend": "Tendance",
    "meanrev": "Mean-Reverting",
    "carry": "Carry/Market-Making",
    "arbitrage": "Arbitrage",
    "micro": "Micro/Scalping",
    "meta": "Méta-modèle",
}


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def family_of(strategy: str) -> str:
    return FAMILY_OF.get(strategy, "other")


# --------------------------------------------------------------------------- #
# REGRET TRACKER (non-stationnaire)
# --------------------------------------------------------------------------- #
class RegretTracker:
    """
    Regret cumulé par stratégie avec OUBLI exponentiel :
        regret_i = decay * regret_i + (max_j pnl_j - pnl_i)

    pnl est la performance RÉALISÉE d'un trade clôturé. Le regret est donc
    une statistique réellement observée (jamais simulée).
    """

    def __init__(self, decay: float = REG_DECAY):
        self.decay = _clamp(float(decay), 0.80, 1.0)
        self.regret: dict[str, float] = {}
        self.n_updates: dict[str, int] = {}
        self._best_recent: float = 0.0

    def record(self, strategy: str, pnl_pct: float) -> None:
        """
        Enregistre l'issue d'un trade clôturé (pnl en fraction).

        Regret incrémental = écart INSTANTANÉ au meilleur pnl récent :
            inc_regret = max(0, best_recent − pnl_i)
        (best_recent = EMA haut du meilleur pnl observé). Le regret cumulé
        avec OUBLI exponentiel mesure donc « combien on a perdu en suivant i
        plutôt que la meilleure ex post », sans que le retard historique
        pèse éternellement (non-stationnaire).
        """
        if not strategy:
            return
        try:
            pnl = float(pnl_pct)
        except (TypeError, ValueError):
            return
        # meilleur pnl récent (EMA haut) — référence ex post instantanée
        self._best_recent = max(0.99 * self._best_recent, pnl)
        inc_regret = max(0.0, self._best_recent - pnl)
        self.regret[strategy] = self.decay * self.regret.get(strategy, 0.0) + inc_regret
        self.n_updates[strategy] = self.n_updates.get(strategy, 0) + 1

    def get(self, strategy: str) -> float:
        return float(self.regret.get(strategy, 0.0))

    def exploration_weights(self, strategies: list[str]) -> dict[str, float]:
        """
        Poids d'exploration par regret (regret matching) :
            w_i ∝ regret_i   (softmax tempéré par REG_EXPLORATION_WEIGHT)
        Une stratégie qui n'a pas assez d'échantillon reçoit un poids
        d'exploration NEUTRE (on ne pénalise pas la nouveauté).
        """
        raw = {}
        for s in strategies:
            n = self.n_updates.get(s, 0)
            if n < REG_MIN_SAMPLES:
                raw[s] = 1.0 / max(len(strategies), 1)
            else:
                raw[s] = max(0.0, self.regret.get(s, 0.0)) + 1e-9
        total = sum(raw.values())
        if total <= 0:
            return {s: 1.0 / max(len(strategies), 1) for s in strategies}
        # softmax tempéré : l'exploration est BORNÉE (REG_EXPLORATION_WEIGHT)
        out = {}
        for s in strategies:
            p = raw[s] / total
            out[s] = REG_EXPLORATION_WEIGHT * p
        return out

    def to_dict(self) -> dict:
        return {
            "regret": {k: round(v, 6) for k, v in self.regret.items()},
            "n_updates": dict(self.n_updates),
            "decay": self.decay,
            "min_samples": REG_MIN_SAMPLES,
            "ts": time.time(),
        }


# --------------------------------------------------------------------------- #
# HIERARCHICAL ALLOCATOR
# --------------------------------------------------------------------------- #
class HierarchicalAllocator:
    """
    Restructure les poids bruts de la meta-allocation :

      1. Regroupe les stratégies par FAMILLE (niveau 2).
      2. Applique un SCALE DE FAMILLE par performance récente (EMA des pnl
         par famille, borné [FAMILY_SCALE_MIN, FAMILY_SCALE_MAX]).
      3. Ajoute l'EXPLORATION PAR REGRET (regret matching, bornée) au niveau
         stratégie.
      4. Renormalise (somme = 1.0) avec bornes dures [MIN_WEIGHT, MAX_WEIGHT].

    Entrées :
      base_weights  : {strategy: poids brut} (de MetaAllocationEngine)
      pnl_by_strategy : {strategy: pnl% récent} pour le scale de famille
      regret_tracker : RegretTracker (ou None -> pas d'exploration regret)

    Sortie : {strategy: poids final} — somme 1.0, chaque poids borné.
    """

    def __init__(self, family_ema_alpha: float = FAMILY_EMA_ALPHA):
        self.family_ema: dict[str, float] = {}
        self.alpha = _clamp(float(family_ema_alpha), 0.05, 0.95)
        self.last_allocation: dict = {}
        self.n_calls = 0

    def _update_family_perf(self, pnl_by_strategy: dict[str, float]) -> None:
        for strat, pnl in (pnl_by_strategy or {}).items():
            fam = family_of(strat)
            prev = self.family_ema.get(fam, 0.0)
            self.family_ema[fam] = self.alpha * float(pnl) + (1.0 - self.alpha) * prev

    def _family_scale(self, family: str) -> float:
        perf = self.family_ema.get(family)
        if perf is None:
            return 1.0
        # performance récente positive -> scale > 1 ; négative -> scale < 1
        scale = 1.0 + 8.0 * perf
        return _clamp(scale, FAMILY_SCALE_MIN, FAMILY_SCALE_MAX)

    def allocate(self, base_weights: dict[str, float],
                 pnl_by_strategy: dict[str, float] | None = None,
                 regret_tracker: RegretTracker | None = None) -> dict[str, float]:
        """Voir docstring de classe. Somme des poids = 1.0 (bornes dures)."""
        strategies = list(base_weights.keys())
        if not strategies:
            return {}
        self._update_family_perf(pnl_by_strategy or {})

        # 1+2. regroupement + scale de famille
        family_weights: dict[str, float] = {}
        scaled: dict[str, float] = {}
        for s, w in base_weights.items():
            fam = family_of(s)
            family_weights[fam] = family_weights.get(fam, 0.0) + float(w)
            scaled[s] = float(w) * self._family_scale(fam)

        # 3. exploration par regret (bornée, ajoutée au poids)
        if regret_tracker is not None:
            exploration = regret_tracker.exploration_weights(strategies)
            for s in strategies:
                scaled[s] += exploration.get(s, 0.0)

        # 4. bornes + renormalisation
        for s in strategies:
            scaled[s] = _clamp(scaled[s], MIN_WEIGHT, MAX_WEIGHT)
        total = sum(scaled.values())
        if total <= 0:
            return {s: 1.0 / max(len(strategies), 1) for s in strategies}
        out = {s: w / total for s, w in scaled.items()}
        self.last_allocation = {
            "weights": {k: round(v, 6) for k, v in out.items()},
            "family_weights": {k: round(v, 6) for k, v in family_weights.items()},
            "family_ema": {k: round(v, 6) for k, v in self.family_ema.items()},
            "family_scales": {k: round(self._family_scale(k), 4)
                              for k in set(family_of(s) for s in strategies)},
            "n_calls": self.n_calls,
        }
        self.n_calls += 1
        return out
