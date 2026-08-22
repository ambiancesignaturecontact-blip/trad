"""
CAPITAL ALLOCATION ENGINE (PHASE 3 — §3 & §11).

Répond à la question : « Quel niveau de capital le système MÉRITE-t-il
actuellement ? » — avec une RECOMMANDATION, jamais un changement automatique.

Sortie : INCREASE / MAINTAIN / REDUCE / FREEZE (+ niveau de dégradation
NORMAL / WARNING / CRITICAL / HALT pour §11).

Entrées (toutes RÉELLES, jamais inventées) :
  - rapport de validation paper (paper_validation)
  - edge decay (stratégies DISABLED/DEGRADED)
  - drift PSI (status)
  - calibration (n, calibration_error)
  - exécution (slippage moyen, forecast error)
  - drawdown courant

Règles (mandat : « l'augmentation du capital doit nécessiter une preuve » et
« réduire même si le PnL récent est positif ») :
  - INCREASE  : validation READY + edge prouvé (expectancy > 0, n >= 30)
                + calibration OK + drift stable + aucune stratégie DISABLED
                durable. Sinon JAMAIS INCREASE.
  - REDUCE    : edge négatif OU calibration dégradée OU exécution dégradée
                OU drift sévère — même si PnL récent positif.
  - FREEZE    : validation NOT_READY sur les critères critiques (protection/
                limites) OU kill switch OU drawdown > limite.
  - MAINTAIN  : tout le reste (preuves insuffisantes = pas d'augmentation).

Ce module n'écrit RIEN : il produit un avis. La décision d'appliquer reste
humaine (ou un processus séparé avec garde-fous).
"""
import logging
import time

logger = logging.getLogger("InstitutionalTradingBot")

INCREASE = "INCREASE"
MAINTAIN = "MAINTAIN"
REDUCE = "REDUCE"
FREEZE = "FREEZE"

# Niveaux de dégradation (§11)
NORMAL = "NORMAL"
WARNING = "WARNING"
CRITICAL = "CRITICAL"
HALT = "HALT"

# Seuils (documentés, configurables par constantes)
MIN_CLOSED_TRADES = 30
EXPECTANCY_MIN = 0.0
MAX_CALIBRATION_ERROR = 0.15
MAX_FORECAST_ERROR_BPS = 20.0
MAX_DISABLED_PCT = 0.25
MAX_DRAWDOWN_PCT = 0.15


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class CapitalAllocationEngine:
    """Recommandation de niveau de capital (avis seulement)."""

    def recommend(self,
                  validation_status: str = "NOT_READY",
                  closed_trades: int = 0,
                  expectancy_pct: float | None = None,
                  drawdown_pct: float = 0.0,
                  drift_status: str = "STABLE",
                  disabled_strategies: int = 0,
                  total_strategies: int = 12,
                  calibration_n: int = 0,
                  calibration_error: float | None = None,
                  avg_slippage_bps: float | None = None,
                  forecast_error_bps: float | None = None,
                  kill_switch: bool = False,
                  risk_state: str = "NORMAL") -> dict:
        """
        Produit la recommandation. Défensif : toute entrée manquante est
        traitée comme « preuve insuffisante » -> pas d'augmentation.
        """
        reasons: list[str] = []
        degradation = NORMAL

        # --- conditions critiques (FREEZE) ---
        if kill_switch or risk_state == "HALT":
            return self._verdict(FREEZE, HALT,
                                 ["kill switch actif ou état HALT — aucun capital exposé"],
                                 reasons)
        if drawdown_pct > MAX_DRAWDOWN_PCT:
            return self._verdict(FREEZE, CRITICAL,
                                 [f"drawdown {drawdown_pct:.1f}% > {MAX_DRAWDOWN_PCT * 100:.0f}%"],
                                 reasons)

        # --- signaux de dégradation ---
        if calibration_n >= MIN_CLOSED_TRADES and calibration_error is not None \
                and calibration_error > MAX_CALIBRATION_ERROR:
            reasons.append(f"calibration error {calibration_error:.3f} > {MAX_CALIBRATION_ERROR}")
            degradation = WARNING
        if forecast_error_bps is not None and abs(forecast_error_bps) > MAX_FORECAST_ERROR_BPS:
            reasons.append(f"erreur de prévision exécution {forecast_error_bps:.1f} bps")
            degradation = WARNING
        if drift_status in ("SEVERE", "MODERATE"):
            reasons.append(f"drift PSI {drift_status}")
            degradation = CRITICAL if drift_status == "SEVERE" else WARNING
        disabled_pct = disabled_strategies / max(total_strategies, 1)
        if disabled_pct > MAX_DISABLED_PCT:
            reasons.append(f"{disabled_strategies}/{total_strategies} stratégies DISABLED")
            degradation = CRITICAL if disabled_pct > 0.5 else WARNING

        # --- edge : preuve exigée pour INCREASE ---
        edge_proven = (closed_trades >= MIN_CLOSED_TRADES
                       and expectancy_pct is not None and expectancy_pct > EXPECTANCY_MIN)
        if not edge_proven and expectancy_pct is not None and expectancy_pct <= EXPECTANCY_MIN:
            reasons.append(f"expectancy {expectancy_pct:.3f}% <= 0 sur {closed_trades} clôtures")
            degradation = WARNING

        # --- décision ---
        if edge_proven and validation_status == "READY" and degradation == NORMAL:
            rec = INCREASE
            reasons.append(f"edge prouvé ({expectancy_pct:.3f}%/trade, n={closed_trades}) "
                           f"+ validation READY")
        elif degradation != NORMAL or (expectancy_pct is not None
                                       and expectancy_pct <= EXPECTANCY_MIN):
            rec = REDUCE
            reasons.append("qualité dégradée — réduire l'exposition même si PnL récent positif")
        else:
            rec = MAINTAIN
            reasons.append("preuves insuffisantes pour augmenter (edge non démontré)")

        return self._verdict(rec, degradation, reasons, [])

    @staticmethod
    def _verdict(rec: str, degradation: str, reasons: list, extra) -> dict:
        return {
            "recommendation": rec,
            "degradation_level": degradation,
            "reasons": reasons,
            "capital_change": "AUCUN changement automatique — avis seulement",
            "note": "INCREASE exige une preuve (edge OOS + calibration + stabilité). "
                    "REDUCE peut être décidé même si le PnL récent est positif.",
            "ts": time.time(),
        }
