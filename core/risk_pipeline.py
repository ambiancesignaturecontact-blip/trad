"""
PIPELINE DE RISQUE UNIFIÉ — LOT 2 (PROMPT MAÎTRE, Faille 3 + Piliers F & G).

Ce module est la SOURCE UNIQUE DE VÉRITÉ du risque. Il corrige :
  • RR INCOHÉRENT  : sizing Kelly à RR=1.5 vs stops réels à RR=1.75–2.0
      -> UNE seule constante REWARD_RISK_RATIO utilisée partout (sizing,
         SL/TP position_manager, backtest).
  • win_rate=0.55 en dur -> Kelly DYNAMIQUE sur le win rate réel par
      stratégie (plancher 0.45 / plafond 0.65 / lissage EMA).
  • Pipeline en désordre dans main.py -> UN SEUL pipeline ordonné et testé
      (RISK_PIPELINE_ORDER ci-dessous, chaque étape journalisée).
  • Pas de HALT sur choc -> machine à états NORMAL / CAUTION / HALT avec
      cool-down et redémarrage progressif (mentalité n°1 : survivre d'abord ;
      n°2 : l'edge est net des coûts ; n°7 : l'asymétrie est la loi ;
      n°14 : penser en probabilités).
"""
import logging
import time
from typing import Dict, List, Optional, Tuple

from core.config import settings

logger = logging.getLogger("RiskPipeline")

# --------------------------------------------------------------------------- #
# SOURCE UNIQUE DE VÉRITÉ (Faille 3 : RR unifié)
# --------------------------------------------------------------------------- #
# Reward/Risk utilisé PARTOUT : sizing Kelly + SL/TP position_manager + backtest.
# Aligné sur les stops réels (anciennement RR=1.75–2.0) => 1.8 par défaut.
REWARD_RISK_RATIO: float = settings.get_float("risk", "reward_risk_ratio", 1.8)

# RR minimal pour entrer en position (Pilier F, exigence 3) — « je ne trade que
# si RR >= X ». En forte volatilité, un RR SUPÉRIEUR est exigé (exigence 4).
MIN_REWARD_RISK: float = settings.get_float("risk", "min_reward_risk", 1.5)
MIN_REWARD_RISK_HIGH_VOL: float = settings.get_float("risk", "min_reward_risk_high_vol", 2.0)
HIGH_VOL_RETURN_STD: float = settings.get_float("risk", "high_vol_return_std", 0.02)

# Coût aller-retour estimé (frais taker + slippage moyen), en fraction du capital
ROUND_TRIP_COST_PCT: float = settings.get_float("risk", "round_trip_cost_pct", 0.002)

# Kelly : fraction prudente + bornes du win rate (Pilier F, exigence 1)
KELLY_FRACTION: float = settings.get_float("risk", "kelly_fraction", 0.15)
WIN_RATE_FLOOR: float = 0.45
WIN_RATE_CEIL: float = 0.65

# Stops (Pilier G : plafond dur 25 %/actif déjà via max_per_asset_pct)
STOP_LOSS_PCT: float = settings.get_float("risk", "stop_loss_pct", 0.03)
ATR_MULT_SL: float = settings.get_float("risk", "atr_mult_sl", 2.0)

# Machine à états (Pilier G : cool-down + redémarrage progressif)
HALT_COOLDOWN_MINUTES: float = settings.get_float("risk", "halt_cooldown_minutes", 15.0)
# Étapes de redémarrage : (facteur de taille, minutes depuis le début du restart)
RESTART_STAGES: List[Tuple[float, float]] = [
    (0.25, 0.0),    # 25 % immédiatement après le cool-down
    (0.50, 30.0),   # 50 % après 30 min
    (0.75, 60.0),   # 75 % après 1 h
    (1.00, 120.0),  # retour à 100 % après 2 h stables
]

# Ordre OFFICIEL des multiplicateurs du pipeline (Pilier G, exigence 1).
# Les plafonds DURS (min) passent en premier, puis les overlays (mul).
RISK_PIPELINE_ORDER: List[str] = [
    "cvar_cap",        # 1. CVaR portfolio (max perte 2 %)          [min]
    "max_asset_cap",   # 2. Plafond dur 25 %/actif                   [min]
    "conviction",      # 3. Intensité du signal (0..1)               [mul]
    "risk_state",      # 4. État NORMAL/CAUTION/HALT                 [mul]
    "news_shock",      # 5. Choc d'actualité systémique              [mul]
    "macro_event",     # 6. Événement macro réel proche              [mul]
    "macro_tactile",   # 7. Override humain (boutons Telegram)       [mul]
    "onchain",         # 8. Risque on-chain réel                     [mul]
    "correlation",     # 9. Concentration de corrélation             [mul]
    "regime_confidence",# 9bis. Certitude du régime + causalité        [mul]
    "capacity",        # 9ter. Capacité (1% volume 24h, Pilier L)     [mul]
    "cash_reserve",    # 9quater. Réserve cash (jamais 100% investi)  [mul]
    "order_flow",      # 10. Flux toxique (delta/CVD/VPIN/OFI)       [mul]
    "confidence",      # 11. Indice de confiance (méta-cognition)    [mul]
    "organization",    # 12. Facteur desk (organisation)             [mul]
    "rlhf",            # 13. Modulateur préférences humaines         [mul]
    "vol_targeting",   # 14. Volatilité cible (vol targeting)        [mul]
    "tradability",     # 15. Slippage attendu / tradabilité         [mul]
]


# --------------------------------------------------------------------------- #
# KELLY DYNAMIQUE (Pilier F, exigence 1)
# --------------------------------------------------------------------------- #
def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def smoothed_win_rate(previous: Optional[float], outcome: float, alpha: float = 0.25) -> float:
    """
    Lissage EMA du win rate (Pilier F : éviter les oscillations dues au faible
    échantillon). Sans historique, on part d'un a priori NEUTRE de 0.50
    (mentalité n°14 : penser en probabilités, pas en certitudes).
    """
    prior = previous if previous is not None else 0.50
    return alpha * outcome + (1.0 - alpha) * prior


def kelly_dynamic(win_rate: Optional[float],
                  reward_risk: float = REWARD_RISK_RATIO,
                  fraction: float = KELLY_FRACTION) -> float:
    """
    Kelly fractionnel DYNAMIQUE, NET de coûts (mentalité n°2).
    - win_rate None ou hors bornes -> plancher 0.45 / plafond 0.65 (exigence 1)
    - R réduit du coût aller-retour pour ne pas surévaluer l'edge
    - fraction conservatrice (15 %) pour survivre d'abord (mentalité n°1)
    """
    p = clamp(float(win_rate) if win_rate is not None else WIN_RATE_FLOOR,
              WIN_RATE_FLOOR, WIN_RATE_CEIL)
    R_net = max(reward_risk - ROUND_TRIP_COST_PCT, 0.01)
    kelly = (p * R_net - (1.0 - p)) / R_net
    return max(0.0, kelly * fraction)


# --------------------------------------------------------------------------- #
# RR ADAPTATIF + ASYMÉTRIE VS COÛTS (Pilier F, exigences 4 & 5)
# --------------------------------------------------------------------------- #
def rr_requirement(regime_id: Optional[int] = None,
                   vol_mean: Optional[float] = None) -> float:
    """
    RR minimal exigé, ADAPTATIF : en forte volatilité (régime bear high vol
    (1), erratic (3), ou écart-type récent élevé), on exige un RR SUPÉRIEUR —
    l'incertitude est plus grande, un RR fixe serait insuffisant.
    """
    if regime_id in (1, 3):
        return MIN_REWARD_RISK_HIGH_VOL
    if vol_mean is not None and vol_mean >= HIGH_VOL_RETURN_STD:
        return MIN_REWARD_RISK_HIGH_VOL
    return MIN_REWARD_RISK


def rr_net_positive(reward_risk: float, sl_distance_pct: float,
                    cost_pct: float = ROUND_TRIP_COST_PCT) -> bool:
    """
    Asymétrie explicite (exigence 5) : le RR brut n'a de valeur que NET des
    coûts. Espérance de gain nette par unité de risque :
        (RR - 1) × distance_SL > coûts aller-retour
    Sinon l'espérance est négative même avec un bon RR brut.
    """
    if reward_risk <= 1.0 or sl_distance_pct <= 0:
        return False
    return (reward_risk - 1.0) * sl_distance_pct > cost_pct


def entry_rr_filter(reward_risk: float, regime_id: Optional[int] = None,
                    vol_mean: Optional[float] = None,
                    sl_distance_pct: Optional[float] = None,
                    cost_pct: float = ROUND_TRIP_COST_PCT) -> Tuple[bool, str]:
    """
    Filtre d'entrée « RR minimal » (exigence 3) :
     1. RR configuré >= RR requis (adaptatif au régime/volatilité)
     2. Asymétrie nette positive : (RR-1)×SL > coûts
    Retourne (autorisé, raison). Si SL inconnue -> on utilise le défaut
    prudent (stop % configuré) pour ne jamais entrer sans vérification.
    """
    required = rr_requirement(regime_id, vol_mean)
    if reward_risk < required:
        return False, (f"RR {reward_risk:.2f} < requis {required:.2f} "
                       f"(régime {regime_id}, vol {vol_mean})")
    sl = sl_distance_pct if sl_distance_pct and sl_distance_pct > 0 else STOP_LOSS_PCT
    if not rr_net_positive(reward_risk, sl, cost_pct):
        return False, (f"Asymétrie nette négative : (RR-1)×SL "
                       f"({(reward_risk-1)*sl:.4f}) <= coûts ({cost_pct:.4f})")
    return True, "RR conforme"


# --------------------------------------------------------------------------- #
# MACHINE À ÉTATS NORMAL / CAUTION / HALT (Faille 3 + Pilier I)
# --------------------------------------------------------------------------- #
class RiskStateMachine:
    """
    Machine à états du risque. NORMAL -> CAUTION -> HALT.
    - HALT : AUCUN nouvel ordre (la protection des positions existantes reste
      active — atomicité, Pilier G).
    - Cool-down : après un HALT, on attend HALT_COOLDOWN_MINUTES avant de
      redémarrer.
    - Redémarrage PROGRESSIF par étapes (25 % -> 50 % -> 75 % -> 100 %),
      jamais de retour brutal à 100 % (Pilier G, exigence 4).
    """

    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    HALT = "HALT"

    def __init__(self, cooldown_minutes: float = HALT_COOLDOWN_MINUTES,
                 restart_stages: List[Tuple[float, float]] = None):
        self.cooldown_seconds = cooldown_minutes * 60.0
        self.restart_stages = restart_stages or RESTART_STAGES
        self.state = self.NORMAL
        self.reason = ""
        self.entered_ts = 0.0
        self.restart_ts = 0.0       # début du redémarrage progressif
        self.prev_state = self.NORMAL
        self._alerted_transition = False

    # -- transitions -------------------------------------------------- #
    def enter(self, level: str, reason: str) -> bool:
        """Passe à CAUTION ou HALT. Retourne True si l'état a changé."""
        if level == self.HALT and self.state != self.HALT:
            self.state = self.HALT
            self.reason = reason
            self.entered_ts = time.time()
            self.restart_ts = 0.0
            logger.critical(f"🚨 RISK STATE -> HALT ({reason})")
            return True
        if level == self.CAUTION and self.state == self.NORMAL:
            self.state = self.CAUTION
            self.reason = reason
            self.entered_ts = time.time()
            logger.warning(f"⚠️ RISK STATE -> CAUTION ({reason})")
            return True
        if level == self.CAUTION and self.state == self.CAUTION:
            # mise à jour de la raison (sans changement d'état)
            self.reason = reason
        return False

    def reset(self, reason: str = "opérateur") -> bool:
        """Retour manuel à NORMAL (endpoint / Telegram)."""
        if self.state != self.NORMAL:
            self.state = self.NORMAL
            self.reason = f"reset({reason})"
            self.restart_ts = 0.0
            logger.info(f"🟢 RISK STATE -> NORMAL (reset {reason})")
            return True
        return False

    # -- taille autorisée --------------------------------------------- #
    def scale_factor(self) -> float:
        """Facteur multiplicatif de taille selon l'état courant."""
        if self.state == self.HALT:
            return 0.0
        if self.state == self.CAUTION:
            return self._caution_stage_factor()
        return 1.0

    def _caution_stage_factor(self) -> float:
        """En CAUTION issu d'un HALT : redémarrage progressif par étapes."""
        if self.restart_ts <= 0:
            return 0.5  # CAUTION direct (choc léger) : -50 %
        elapsed = (time.time() - self.restart_ts) / 60.0
        factor = 0.25
        for f, minutes in self.restart_stages:
            if elapsed >= minutes:
                factor = f
        return factor

    # -- gestion du temps (appelée à chaque tick) ---------------------- #
    def tick(self, now: float = None) -> bool:
        """
        À appeler à chaque itération de la boucle. Gère le cool-down du HALT
        puis la progression des étapes de redémarrage. Retourne True si l'état
        a changé (pour alerter Telegram).
        """
        now = now or time.time()
        changed = False
        if self.state == self.HALT:
            if now - self.entered_ts >= self.cooldown_seconds:
                self.state = self.CAUTION
                self.reason = f"redémarrage progressif (après HALT: {self.reason})"
                self.restart_ts = now
                logger.warning(f"🟠 RISK STATE -> CAUTION (cool-down écoulé, redémarrage progressif)")
                changed = True
        elif self.state == self.CAUTION and self.restart_ts > 0:
            elapsed = (now - self.restart_ts) / 60.0
            max_minutes = max((m for _, m in self.restart_stages), default=0.0)
            if elapsed >= max_minutes:
                self.state = self.NORMAL
                self.reason = "redémarrage terminé (2h stables)"
                self.restart_ts = 0.0
                logger.info("🟢 RISK STATE -> NORMAL (redémarrage progressif terminé)")
                changed = True
        return changed

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "reason": self.reason,
            "entered_ts": self.entered_ts,
            "scale_factor": self.scale_factor(),
            "restart_ts": self.restart_ts,
        }


# --------------------------------------------------------------------------- #
# SUIVI DU WIN RATE RÉEL PAR STRATÉGIE (Pilier F, exigence 1)
# --------------------------------------------------------------------------- #
class StrategyWinRateTracker:
    """
    Calcule le win rate RÉEL par stratégie à partir des trades clôturés
    (STATE["strategy_win_rates"] + STATE["strategy_trade_counts"]), avec
    lissage EMA. C'est LA donnée qui alimente le Kelly dynamique ET le
    filtre méta-label (López de Prado) — plus jamais de 0.55 en dur.
    """

    def __init__(self, state: dict, alpha: float = 0.25):
        self.state = state
        self.state.setdefault("strategy_win_rates", {})
        self.state.setdefault("strategy_trade_counts", {})
        self.alpha = alpha

    def record(self, strategy: str, pnl_pct: float) -> None:
        """Enregistre l'issue d'un trade clôturé (gagnant si pnl_pct > 0)."""
        if not strategy:
            return
        rates = self.state["strategy_win_rates"]
        counts = self.state["strategy_trade_counts"]
        outcome = 1.0 if pnl_pct > 0 else 0.0
        rates[strategy] = smoothed_win_rate(rates.get(strategy), outcome, self.alpha)
        counts[strategy] = counts.get(strategy, 0) + 1

    def get(self, strategy: str) -> float:
        """Win rate lissée et bornée (0.45..0.65) pour le Kelly dynamique."""
        wr = self.state["strategy_win_rates"].get(strategy)
        if wr is None:
            return WIN_RATE_FLOOR  # a priori prudent sans historique
        return clamp(float(wr), WIN_RATE_FLOOR, WIN_RATE_CEIL)

    def samples(self, strategy: str) -> int:
        return int(self.state["strategy_trade_counts"].get(strategy, 0))


# --------------------------------------------------------------------------- #
# PIPELINE UNIFIÉ (Pilier G, exigence 1) — ordre documenté + tracé
# --------------------------------------------------------------------------- #
def apply_risk_pipeline(base_qty: float,
                        cvar_qty: float,
                        max_asset_qty: float,
                        conviction: float,
                        risk_state_scale: float,
                        order_flow_scale: float = 1.0,
                        regime_confidence_scale: float = 1.0,
                        capacity_scale: float = 1.0,
                        cash_reserve_scale: float = 1.0,
                        news_scale: float = 1.0,
                        macro_scale: float = 1.0,
                        tactile_scale: float = 1.0,
                        onchain_scale: float = 1.0,
                        corr_scale: float = 1.0,
                        confidence_scale: float = 1.0,
                        org_scale: float = 1.0,
                        rlhf_scale: float = 1.0,
                        vol_scale: float = 1.0,
                        tradability_scale: float = 1.0) -> dict:
    """
    Applique TOUS les facteurs de risque dans l'ORDRE documenté
    (RISK_PIPELINE_ORDER). Chaque étape est tracée pour l'audit et la
    télémétrie. Comportement par défaut PRUDENT : en cas de doute, réduire.
    """
    steps: List[Dict] = []
    qty = float(base_qty)

    # -- plafonds durs (min) -- #
    qty = min(qty, cvar_qty)
    steps.append({"step": "cvar_cap", "op": "min", "value": cvar_qty,
                  "qty_after": qty})
    qty = min(qty, max_asset_qty)
    steps.append({"step": "max_asset_cap", "op": "min", "value": max_asset_qty,
                  "qty_after": qty})

    # -- overlays multiplicatifs, dans l'ordre -- #
    factors = {
        "conviction": clamp(float(conviction), 0.0, 1.0),
        "risk_state": max(0.0, float(risk_state_scale)),
        "order_flow": max(0.0, float(order_flow_scale)),
        "regime_confidence": max(0.0, float(regime_confidence_scale)),
        "capacity": max(0.0, float(capacity_scale)),
        "cash_reserve": max(0.0, float(cash_reserve_scale)),
        "news_shock": max(0.0, float(news_scale)),
        "macro_event": max(0.0, float(macro_scale)),
        "macro_tactile": max(0.0, float(tactile_scale)),
        "onchain": max(0.0, float(onchain_scale)),
        "correlation": max(0.0, float(corr_scale)),
        "confidence": max(0.0, float(confidence_scale)),
        "organization": max(0.0, float(org_scale)),
        "rlhf": max(0.0, float(rlhf_scale)),
        "vol_targeting": max(0.0, float(vol_scale)),
        "tradability": max(0.0, float(tradability_scale)),
    }
    for name in RISK_PIPELINE_ORDER[2:]:
        f = factors.get(name, 1.0)
        qty *= f
        steps.append({"step": name, "op": "mul", "value": f, "qty_after": qty})

    final_scale = qty / base_qty if base_qty > 0 else 0.0
    return {
        "qty": max(0.0, qty),
        "final_scale": final_scale,
        "steps": steps,
    }
