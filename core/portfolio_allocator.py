"""
PORTFOLIO ALLOCATOR — ALLOCATION TOP-DOWN EN CASCADE (PROMPT MAÎTRE, Pilier L).

Un expert n°1 pense en PORTEFEUILLE, pas en trades isolés. Les briques
existaient (HRP, optimiseur multi-objectif, allocateur dynamique) mais
n'étaient PAS orchestrées. Ce module impose la hiérarchie :

  (a) BUDGET DE RISQUE TOTAL  : vol cible + CVaR portfolio -> capital
       investissable (jamais plus de (1 - réserve cash)).
  (b) ALLOCATION DESKS/STRATÉGIES : risk parity / HRP sur la corrélation
       RÉELLE + pénalité des stratégies REDONDANTES (corrélation élevée
       entre stratégies = concentre le risque sans diversifier).
  (c) SIZING PAR TRADE : le sizing existant (LOT 2) applique le budget.

Exigences couvertes :
  - Réserve de cash OBLIGATOIRE (10-20 %) : jamais 100 % investi.
  - Rebalancing périodique vers les cibles (anti-drift).
  - Capacité (capacity) : plafond = participation raisonnable du volume réel.
"""
import logging
import time

import numpy as np

from core.config import settings

logger = logging.getLogger("PortfolioAllocator")

# P1-8 (audit §3) : constantes branchées sur core/config.py (config.yaml).
# Les défauts restent STRICTEMENT identiques aux valeurs historiques — un
# opérateur peut maintenant les surcharger sans toucher au code.
CASH_RESERVE_PCT = settings.get_float("portfolio", "cash_reserve_pct", 0.15)
TARGET_VOL_ANNUAL = settings.get_float("portfolio", "target_vol_annual", 0.10)
REBALANCE_HOURS = settings.get_float("portfolio", "rebalance_hours", 24.0)
MAX_PARTICIPATION_PCT = settings.get_float("portfolio", "max_participation_pct", 0.01)
REDUNDANT_CORR = settings.get_float("portfolio", "redundant_corr", 0.85)


class PortfolioAllocator:
    """Orchestrateur top-down du portefeuille (Pilier L)."""

    def __init__(self, cash_reserve_pct: float = CASH_RESERVE_PCT,
                 target_vol_annual: float = TARGET_VOL_ANNUAL,
                 rebalance_hours: float = REBALANCE_HOURS):
        self.cash_reserve_pct = cash_reserve_pct
        self.target_vol_annual = target_vol_annual
        self.rebalance_hours = rebalance_hours
        self.last_rebalance_ts = 0.0
        self.allocation: dict = {"status": "PENDING"}

    # ------------------------------------------------------------------ #
    # (a) BUDGET DE RISQUE TOTAL
    # ------------------------------------------------------------------ #
    def total_risk_budget(self, total_capital: float,
                          portfolio_cvar_pct: float | None = None,
                          realized_vol_annual: float | None = None) -> dict:
        """
        Budget de risque TOTAL :
        1. Réserve de cash OBLIGATOIRE : investissable = capital × (1 - réserve).
           On ne JAMAIS être investi à 100 % (dislocation = besoin de cash).
        2. Vol cible : si la vol réalisée dépasse la cible, on réduit
           l'exposition (vol targeting).
        3. CVaR : si le CVaR portfolio est élevé, on réduit (garde prudente).
        """
        if total_capital <= 0:
            return {"investable": 0.0, "budget": 0.0, "cash_reserve_pct": self.cash_reserve_pct}
        investable = total_capital * (1.0 - self.cash_reserve_pct)

        vol_scale = 1.0
        if realized_vol_annual and realized_vol_annual > 0:
            vol_scale = max(0.2, min(1.0, self.target_vol_annual / realized_vol_annual))

        cvar_scale = 1.0
        if portfolio_cvar_pct and portfolio_cvar_pct > 0:
            # CVaR > 5 % -> réduction ; CVaR <= 2 % -> pas de pénalité
            cvar_scale = max(0.3, min(1.0, 1.0 - (portfolio_cvar_pct - 0.02) / 0.08))

        budget = investable * vol_scale * cvar_scale
        return {
            "investable": round(investable, 2),
            "budget": round(budget, 2),
            "cash_reserve_pct": self.cash_reserve_pct,
            "cash_reserve_usd": round(total_capital * self.cash_reserve_pct, 2),
            "vol_scale": round(vol_scale, 4),
            "cvar_scale": round(cvar_scale, 4),
        }

    # ------------------------------------------------------------------ #
    # (b) ALLOCATION STRATÉGIES + DIVERSIFICATION RÉELLE
    # ------------------------------------------------------------------ #
    def strategy_diversification(self, strategy_returns: dict[str, list[float]],
                                 min_samples: int = 20) -> dict:
        """
        Diversification RÉELLE : mesure la corrélation entre STRATÉGIES (pas
        seulement entre actifs) et PÉNALISE les stratégies redondantes qui
        concentrent le risque sans diversifier (PDF Pilier L).

        Retourne {strategy: {"weight_penalty": float, "max_corr": float}}.
        Sans échantillon suffisant -> aucune pénalité (neutre, jamais de
        signal fabriqué).
        """
        names = [s for s, r in strategy_returns.items() if len(r) >= min_samples]
        if len(names) < 2:
            return {s: {"weight_penalty": 1.0, "max_corr": 0.0} for s in strategy_returns}

        matrix = np.array([strategy_returns[n][-min_samples:] for n in names])
        # FIX (logs) : une série à variance nulle (rendements constants)
        # produit des NaN dans np.corrcoef (RuntimeWarning + corr invalide).
        # On remplace les NaN par 0 (aucune corrélation mesurable = neutre).
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(matrix)
            corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 0.0)

        result = {}
        for i, n in enumerate(names):
            max_corr = float(np.max(np.abs(corr[i])))
            # pénalité : corrélation > seuil -> poids réduit (anti-redondance)
            if max_corr > REDUNDANT_CORR:
                penalty = max(0.3, 1.0 - (max_corr - REDUNDANT_CORR) * 2.0)
            else:
                penalty = 1.0
            result[n] = {"weight_penalty": round(penalty, 4),
                         "max_corr": round(max_corr, 4)}
        # stratégies sans échantillon : pas de pénalité
        for s in strategy_returns:
            result.setdefault(s, {"weight_penalty": 1.0, "max_corr": 0.0})
        return result

    # ------------------------------------------------------------------ #
    # (c) SIZING PAR TRADE — capacité + facteur d'exposition portfolio
    # ------------------------------------------------------------------ #
    def capacity_cap_qty(self, symbol: str, volume_24h: float | None,
                         current_price: float, participation_pct: float = MAX_PARTICIPATION_PCT
                         ) -> float | None:
        """
        CAPACITÉ (mentalité n°11) : toute stratégie a une taille maximale
        au-delà de laquelle elle se détruit elle-même (impact de marché).
        Plafond = participation_pct (1 %) du volume réel 24h de l'actif.
        Sans volume réel -> None (aucune donnée = pas de plafond fabriqué).
        """
        if not volume_24h or volume_24h <= 0:
            return None
        # volume_24h est en UNITÉS d'actif (ex: BTC) -> cap = participation du
        # volume réel, directement en unités
        return float(volume_24h) * participation_pct

    def portfolio_exposure_factor(self, state: dict, active_balance_key: str,
                                  current_equity: float | None = None) -> float:
        """
        Garde de réserve de cash au niveau PORTEFEUILLE : si l'exposition
        investie approche (1 - réserve), on réduit les nouveaux trades.
        Retourne un facteur 0..1 (1.0 = aucune contrainte).
        """
        try:
            positions = state.get("cached_positions") or []
            invested = 0.0
            for p in positions:
                price = state.get("assets", {}).get(p.get("symbol", ""), {}).get("price")
                if price is None:
                    price = state.get("last_known_prices", {}).get(p.get("symbol"), 0.0)
                if price:
                    invested += float(p.get("qty", 0.0)) * float(price)
            equity = current_equity or state.get("current_equity") or 0.0
            if equity <= 0:
                return 1.0
            exposure = invested / equity
            # P1-14 (audit §2.7) : le plafond TOTAL explicite (config
            # risk.max_exposure_normal, défaut 0.75) complète la réserve cash
            # (1 - cash_reserve) — le plus strict des deux s'applique. Avant,
            # seul cash_reserve bridait (0.85) et max_exposure_normal (0.25)
            # était une constante morte à la même valeur que le plafond par
            # actif : une position de 25 % épuisait « tout » le budget.
            max_exposure = 1.0 - self.cash_reserve_pct
            try:
                max_exposure = min(
                    max_exposure,
                    settings.get_float("risk", "max_exposure_normal", 0.75),
                )
            except Exception:
                pass
            if exposure >= max_exposure:
                return 0.0   # plus de cash disponible -> pas de nouveau trade
            return max(0.0, (max_exposure - exposure) / max_exposure)
        except Exception as e:
            logger.debug(f"Exposure factor error: {e}")
            return 1.0

    # ------------------------------------------------------------------ #
    # REBALANCING
    # ------------------------------------------------------------------ #
    def should_rebalance(self, now: float = None) -> bool:
        now = now or time.time()
        return (now - self.last_rebalance_ts) >= self.rebalance_hours * 3600.0

    def rebalance(self, state: dict, total_capital: float,
                  portfolio_cvar_pct: float | None = None,
                  realized_vol_annual: float | None = None) -> dict:
        """
        Recalcule le budget top-down et le stocke dans STATE pour la
        télémétrie et le sizing (anti-drift : on revient vers les cibles).
        """
        budget = self.total_risk_budget(total_capital, portfolio_cvar_pct,
                                        realized_vol_annual)
        self.allocation = {
            "ts": time.time(),
            "total_risk_budget": budget,
            "cash_reserve_pct": self.cash_reserve_pct,
        }
        self.last_rebalance_ts = time.time()
        state["portfolio_allocation"] = self.allocation
        logger.info(
            f"💼 PORTEFEUILLE: budget investissable {budget['budget']:,.0f}$ "
            f"sur {total_capital:,.0f}$ (réserve cash {self.cash_reserve_pct*100:.0f}%, "
            f"vol scale {budget['vol_scale']:.2f}, cvar scale {budget['cvar_scale']:.2f})")
        return self.allocation

    def to_dict(self) -> dict:
        return self.allocation
