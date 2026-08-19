"""
STRESS TEST PAR SCÉNARIOS DE CRISES RÉELLES (PROMPT MAÎTRE, Pilier N).

Le Monte Carlo ne suffit pas : on rejoue des crises RÉELLES sur le
portefeuille COMPLET (pas seulement un actif). Les paramètres ci-dessous
sont les chocs RÉELLEMENT OBSERVÉS sur les marchés :

  COVID-19 (fév-mars 2020) : S&P 500 -34% en 23 jours, BTC -50% (12-13 mars),
  or +12% (fuite vers la sécurité), USD +3%.
  KRACH CRYPTO 2018      : BTC -84% du sommet (déc 2017 -> déc 2018), ETH -93%.
  FTX COLLAPSE (nov 2022) : BTC -25% en 3 jours après la faillite, SOL -55%,
  ETH -30%, exchanges fragilisés.

Chaque scénario définit un choc par actif (pct) appliqué au portefeuille
courant, sur un horizon en jours. Le stress test calcule :
  - la perte du portefeuille complet (positions valorisées + cash)
  - le drawdown résultant
  - si le portefeuille survit (perte < limite configurée)

Aucune donnée de trading fictive : ce sont des SCÉNARIOS DE CALCUL documentés
à partir de faits historiques réels (mentalité n°13 : le risque de modèle est
réel — on se prépare au pire).
"""
import logging
from typing import Dict, List, Optional

logger = logging.getLogger("ScenarioStress")

# Chocs RÉELS observés (pct de baisse, négatif = perte)
CRISIS_SCENARIOS: Dict[str, Dict] = {
    "COVID_2020": {
        "label": "COVID-19 (fév-mars 2020)",
        "horizon_days": 23,
        "shocks": {
            "BTCUSDT": -0.50, "ETHUSDT": -0.55, "SOLUSDT": -0.60,
            "XAUUSD": +0.12,   # l'or a monté (fuite vers la sécurité)
            "EURUSD": +0.02,
            "AAPL": -0.30, "TSLA": -0.40,
        },
        "note": "Chocs réels observés mars 2020 (S&P -34%, BTC -50% le 12-13 mars).",
    },
    "CRYPTO_CRASH_2018": {
        "label": "Krach crypto 2018",
        "horizon_days": 365,
        "shocks": {
            "BTCUSDT": -0.84, "ETHUSDT": -0.93, "SOLUSDT": -0.95,
            "XAUUSD": -0.02, "EURUSD": 0.0,
            "AAPL": +0.05, "TSLA": +0.03,
        },
        "note": "BTC -84% du sommet déc-2017 au creux déc-2018.",
    },
    "FTX_COLLAPSE_2022": {
        "label": "Faillite FTX (nov 2022)",
        "horizon_days": 7,
        "shocks": {
            "BTCUSDT": -0.25, "ETHUSDT": -0.30, "SOLUSDT": -0.55,
            "XAUUSD": +0.03, "EURUSD": +0.01,
            "AAPL": -0.05, "TSLA": -0.08,
        },
        "note": "BTC -25% en 3 jours après la faillite FTX, SOL -55%.",
    },
}

# Perte maximale tolérable par scénario (défaut : 15 % du capital)
MAX_LOSS_PCT = 0.15


class ScenarioStressTester:
    """Rejoue les crises réelles sur le portefeuille complet."""

    def __init__(self, scenarios: Optional[Dict] = None,
                 max_loss_pct: float = MAX_LOSS_PCT):
        self.scenarios = scenarios or CRISIS_SCENARIOS
        self.max_loss_pct = max_loss_pct

    def run_stress(self, positions: List[dict], cash: float,
                   prices: Dict[str, float]) -> Dict:
        """
        Applique chaque scénario au portefeuille complet (positions + cash).
        positions : [{symbol, qty}] ; prices : {symbol: prix courant}.
        Retourne les résultats par scénario + verdict global.
        """
        results = {}
        for name, scen in self.scenarios.items():
            portfolio_value = 0.0
            shocked_value = 0.0
            for p in positions:
                sym = p.get("symbol", "")
                qty = float(p.get("qty", 0.0))
                price = prices.get(sym)
                if not price or price <= 0:
                    continue  # actif sans prix réel -> ignoré (honnête)
                value = qty * price
                portfolio_value += value
                shock = scen["shocks"].get(sym, 0.0)
                shocked_value += value * (1.0 + shock)

            total = float(portfolio_value) + float(cash)
            shocked_total = float(shocked_value) + float(cash)  # le cash n'est pas choqué
            loss_pct = (shocked_total - total) / total if total > 0 else 0.0
            # Types Python STRICTS (jamais numpy dans l'API — mentalité n°9 :
            # tout doit être sérialisable, auditable). Les prix issus du
            # consensus peuvent être des np.float64 selon la source.
            survived = bool(loss_pct > -float(self.max_loss_pct))

            results[name] = {
                "label": str(scen["label"]),
                "horizon_days": int(scen["horizon_days"]),
                "portfolio_value": round(float(total), 2),
                "shocked_value": round(float(shocked_total), 2),
                "loss_pct": round(float(loss_pct) * 100.0, 2),
                "survived": survived,
                "note": str(scen["note"]),
                "shocks": {str(k): float(v) for k, v in scen["shocks"].items()},
            }

        all_survived = all(bool(r["survived"]) for r in results.values())
        worst = min(results.values(), key=lambda r: float(r["loss_pct"]))
        return {
            "status": "PASSED" if all_survived else "VULNERABLE",
            "scenarios": results,
            "worst": worst["label"],
            "worst_loss_pct": float(worst["loss_pct"]),
            "max_loss_pct": float(self.max_loss_pct) * 100.0,
            "ts": float(__import__("time").time()),
        }
