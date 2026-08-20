"""
RISQUE DE CONTREPARTIE (PROMPT MAÎTRE, Pilier P).

Les cryptos ajoutent un risque que les actions n'ont pas : l'exchange peut
disparaître (FTX). Un expert n°1 le gère EXPLICITEMENT :

  1. Limite de capital PAR exchange : ne jamais laisser plus de X % du capital
     sur un seul exchange (solvabilité / risque de run).
  2. Signaux d'alerte : retraits suspendus, spreads anormaux, volume anormal
     (signaux de run bancaire sur un exchange) -> alerte + réduction.
  3. Garde : hot wallet (petit, trading) vs cold/self-custody (gros, long
     terme) — la clé privée EVM reste chiffrée Fernet (déjà en place).

Aucune donnée fictive : sans donnée réelle, les signaux sont neutres.
"""
import logging
import time
from typing import Dict, Optional

from core.config import settings

logger = logging.getLogger("CounterpartyRisk")

# P1-8 (audit §3) : constantes branchées sur core/config.py (config.yaml).
MAX_CAPITAL_PER_EXCHANGE_PCT = settings.get_float("counterparty", "max_capital_per_exchange_pct", 0.40)
ALERT_SPREAD_BPS = settings.get_float("counterparty", "alert_spread_bps", 25.0)
ALERT_VOLUME_DROP_PCT = settings.get_float("counterparty", "alert_volume_drop_pct", 0.30)


class CounterpartyRiskManager:
    """Surveille le risque de contrepartie par exchange (Pilier P)."""

    def __init__(self, max_capital_per_exchange_pct: float = MAX_CAPITAL_PER_EXCHANGE_PCT):
        self.max_pct = max_capital_per_exchange_pct
        self.alerts: Dict[str, dict] = {}   # exchange -> dernière alerte

    # ------------------------------------------------------------------ #
    # 1. LIMITE DE CAPITAL PAR EXCHANGE
    # ------------------------------------------------------------------ #
    def check_exchange_balance(self, exchange: str, balance_usd: float,
                               total_capital: float) -> dict:
        """
        Vérifie qu'un exchange ne détient pas plus de X % du capital total.
        Retourne {ok, max_allowed, exceed_pct, action}.
        action : "ok" | "warn" (> 90% du plafond) | "block" (> plafond)
        """
        if total_capital <= 0:
            return {"ok": True, "action": "ok", "max_allowed": 0.0}
        max_allowed = total_capital * self.max_pct
        ratio = balance_usd / total_capital if balance_usd > 0 else 0.0
        if balance_usd > max_allowed:
            return {
                "ok": False,
                "action": "block",
                "max_allowed": round(max_allowed, 2),
                "balance": round(balance_usd, 2),
                "exceed_pct": round((ratio - self.max_pct) * 100.0, 2),
                "message": (f"{exchange} détient {ratio*100:.1f}% du capital "
                            f"(plafond {self.max_pct*100:.0f}%) -> réduire l'exposition"),
            }
        if ratio > self.max_pct * 0.9:
            return {
                "ok": True,
                "action": "warn",
                "max_allowed": round(max_allowed, 2),
                "balance": round(balance_usd, 2),
                "exceed_pct": round((ratio - self.max_pct) * 100.0, 2),
                "message": f"{exchange} approche le plafond ({ratio*100:.1f}% / {self.max_pct*100:.0f}%)",
            }
        return {"ok": True, "action": "ok", "max_allowed": round(max_allowed, 2)}

    # ------------------------------------------------------------------ #
    # 2. SIGNAUX D'ALERTE (retraits suspendus, spreads, volume)
    # ------------------------------------------------------------------ #
    def evaluate_exchange_health(self, exchange: str,
                                 spread_bps: Optional[float] = None,
                                 volume_ratio: Optional[float] = None,
                                 withdrawals_suspended: Optional[bool] = None) -> dict:
        """
        Évalue la santé d'un exchange à partir de signaux RÉELS :
          - retraits suspendus (si connu) -> ALERTE CRITIQUE
          - spread anormal (> 25 bps) -> ALERTE
          - volume < 30% de la normale -> ALERTE (liquidité fuyante)
        Retourne un score de risque 0..1 (0 = sain, 1 = critique) + alertes.
        Sans données réelles -> risque 0 (neutre, jamais fabriqué).
        """
        risk = 0.0
        alerts = []
        if withdrawals_suspended:
            risk = 1.0
            alerts.append("RETRAITS SUSPENDUS (signal critique de run)")
        if spread_bps is not None and spread_bps > ALERT_SPREAD_BPS:
            risk = max(risk, 0.7)
            alerts.append(f"spread anormal ({spread_bps:.1f} bps > {ALERT_SPREAD_BPS} bps)")
        if volume_ratio is not None and volume_ratio < ALERT_VOLUME_DROP_PCT:
            risk = max(risk, 0.6)
            alerts.append(f"volume < {ALERT_VOLUME_DROP_PCT*100:.0f}% de la normale (liquidité fuyante)")

        if alerts:
            self.alerts[exchange] = {"ts": time.time(), "risk": risk, "alerts": alerts}
            logger.warning(f"⚠️ CONTREPARTIE {exchange}: risque {risk:.2f} - {', '.join(alerts)}")
        return {"exchange": exchange, "risk_score": round(risk, 2),
                "alerts": alerts, "healthy": risk < 0.5}

    # ------------------------------------------------------------------ #
    # 3. GARDE (HOT vs COLD)
    # ------------------------------------------------------------------ #
    def custody_check(self, hot_balance_usd: float, total_capital: float,
                      cold_threshold_pct: float = 0.7) -> dict:
        """
        Garde : le gros du capital doit rester en COLD/self-custody, le HOT
        (trading) ne doit jamais concentrer plus de (1 - cold_threshold) du
        capital. Mentalité n°1 : si l'exchange disparaît, on ne perd que le
        hot wallet.
        """
        if total_capital <= 0:
            return {"ok": True, "hot_ratio": 0.0, "recommendation": "cold"}
        hot_ratio = hot_balance_usd / total_capital
        max_hot = 1.0 - cold_threshold_pct
        ok = hot_ratio <= max_hot
        return {
            "ok": ok,
            "hot_ratio": round(hot_ratio, 4),
            "max_hot": max_hot,
            "recommendation": "ok" if ok else "TRANSFERER vers cold wallet",
            "message": (f"hot wallet {hot_ratio*100:.1f}% du capital "
                        f"(max {max_hot*100:.0f}%)") if not ok else f"hot wallet {hot_ratio*100:.1f}% (dans les limites)",
        }

    def to_dict(self) -> dict:
        return {
            "max_capital_per_exchange_pct": self.max_pct,
            "recent_alerts": self.alerts,
        }
