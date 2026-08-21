"""
EXECUTION INTELLIGENCE (LOT 7 du mandat).

Consolide la mesure post-trade de la QUALITÉ D'EXÉCUTION :
  - Implementation Shortfall (IS) en bps et en $ : écart fill vs arrival
  - Comparaison PRÉVISION vs RÉALITÉ : expected_slippage (SlippageModel) vs
    realized_slippage (ExecutionAlpha) -> forecast_error
  - Venue quality historique : par venue {n, slippage moyen, latence moyenne,
    fill ratio} — alimente le SOR (pick_best_venue_net) et l'opérateur
  - Latence et fill ratio quand mesurés (None sinon — jamais inventé)

Branchement (main.py) : après chaque fill réel ET papier, une ligne :
    execution_intel.record(venue, symbol, side, qty, arrival, fill, style,
                           expected_bps, realized_bps, latency_ms, fill_ratio, db)

Principes :
  1. JAMAIS bloquant (erreur -> log, le trading continue).
  2. La comparaison prévision/réalité est la clé : un modèle d'exécution qui
     sous-estime systématiquement le slippage doit être VISIBLE et recalibré
     (SlippageModel.recalibrate existe — l'intel alimente le diagnostic).
  3. Persistance best-effort via db.add_event("exec_intel", json) (pas de
     table dédiée : le journal d'événements est déjà ré-écoutable).
  4. DÉMO == RÉAL : aucun flag de mode.
"""
import json
import logging
import time
from collections import deque

logger = logging.getLogger("InstitutionalTradingBot")

MAX_SAMPLES = 2000


class ExecutionIntel:
    def __init__(self, max_samples: int = MAX_SAMPLES):
        self.samples: deque = deque(maxlen=max_samples)
        self.venues: dict[str, dict] = {}      # venue -> stats cumulées
        self.styles: dict[str, dict] = {}      # style -> stats
        self.last = {}

    # ------------------------------------------------------------------ #
    def record(self, venue: str, symbol: str, side: str, qty: float,
               arrival_price: float, fill_price: float, style: str,
               expected_slippage_bps: float | None = None,
               realized_slippage_bps: float | None = None,
               latency_ms: float | None = None,
               fill_ratio: float | None = None,
               db=None) -> None:
        """Enregistre un fill. Calcule IS bps/$ + forecast error. JAMAIS bloquant."""
        try:
            if not venue or arrival_price is None or fill_price is None \
                    or arrival_price <= 0 or fill_price <= 0:
                return
            arrival, fill = float(arrival_price), float(fill_price)
            qty = float(qty) if qty else 0.0

            # IS bps (adverse = positif) — même convention qu'ExecutionAlpha
            if side == "BUY":
                is_bps = (fill - arrival) / arrival * 1e4
            else:
                is_bps = (arrival - fill) / arrival * 1e4
            is_usd = abs(fill - arrival) * qty

            realized = float(realized_slippage_bps) if realized_slippage_bps is not None else is_bps
            expected = float(expected_slippage_bps) if expected_slippage_bps is not None else None
            forecast_error = (realized - expected) if expected is not None else None

            sample = {
                "ts": time.time(), "venue": venue, "symbol": symbol, "side": side,
                "qty": qty, "style": style,
                "is_bps": round(is_bps, 3), "is_usd": round(is_usd, 4),
                "expected_bps": round(expected, 3) if expected is not None else None,
                "realized_bps": round(realized, 3),
                "forecast_error_bps": round(forecast_error, 3) if forecast_error is not None else None,
                "latency_ms": round(float(latency_ms), 2) if latency_ms is not None else None,
                "fill_ratio": round(float(fill_ratio), 4) if fill_ratio is not None else None,
            }
            self.samples.append(sample)
            self.last = sample

            # venue quality (cumul)
            v = self.venues.setdefault(venue, {"n": 0, "is_bps_sum": 0.0,
                                               "latency_sum": 0.0, "latency_n": 0,
                                               "fill_ratio_sum": 0.0, "fill_n": 0,
                                               "forecast_err_sum": 0.0, "fc_n": 0})
            v["n"] += 1
            v["is_bps_sum"] += is_bps
            if latency_ms is not None:
                v["latency_sum"] += float(latency_ms)
                v["latency_n"] += 1
            if fill_ratio is not None:
                v["fill_ratio_sum"] += float(fill_ratio)
                v["fill_n"] += 1
            if forecast_error is not None:
                v["forecast_err_sum"] += forecast_error
                v["fc_n"] += 1

            # par style
            st = self.styles.setdefault(style, {"n": 0, "is_bps_sum": 0.0})
            st["n"] += 1
            st["is_bps_sum"] += is_bps

            # persistance best-effort (journal d'événements ré-écoutable)
            if db is not None:
                try:
                    db.add_event(time.time(), "exec_intel",
                                 json.dumps({k: v for k, v in sample.items()
                                             if v is not None}, default=str))
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"execution_intel record failed ({e})")

    # ------------------------------------------------------------------ #
    def report(self) -> dict:
        """Rapport consolidé : par venue, par style, global, dernière mesure."""
        venues = {}
        for name, v in self.venues.items():
            venues[name] = {
                "n": v["n"],
                "avg_is_bps": round(v["is_bps_sum"] / max(v["n"], 1), 3),
                "avg_latency_ms": round(v["latency_sum"] / max(v["latency_n"], 1), 2)
                if v["latency_n"] else None,
                "avg_fill_ratio": round(v["fill_ratio_sum"] / max(v["fill_n"], 1), 4)
                if v["fill_n"] else None,
                "avg_forecast_error_bps": round(v["forecast_err_sum"] / max(v["fc_n"], 1), 3)
                if v["fc_n"] else None,
            }
        styles = {s: {"n": st["n"], "avg_is_bps": round(st["is_bps_sum"] / max(st["n"], 1), 3)}
                  for s, st in self.styles.items()}
        global_n = sum(v["n"] for v in self.venues.values())
        global_is = sum(v["is_bps_sum"] for v in self.venues.values())
        fc_errs = [v["forecast_err_sum"] / max(v["fc_n"], 1)
                   for v in self.venues.values() if v["fc_n"]]
        return {
            "n": global_n,
            "avg_is_bps": round(global_is / max(global_n, 1), 3) if global_n else None,
            "avg_forecast_error_bps": round(sum(fc_errs) / len(fc_errs), 3) if fc_errs else None,
            "by_venue": venues,
            "by_style": styles,
            "last": self.last,
            "note": "forecast_error = realized − expected (SlippageModel). "
                    "> 0 = le slippage réel dépasse la prévision.",
            "ts": time.time(),
        }
