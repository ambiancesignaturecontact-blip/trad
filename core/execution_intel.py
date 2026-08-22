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


# --------------------------------------------------------------------------- #
# PHASE 3 Cycle 4 — RAPPORT DE FRICTION (chantier slippage p95, observabilité).
# Distribution RÉELLE de la friction d'exécution calculée depuis les fills
# persistés (events paper_fill : slippage_bps, fee, latency) — p50/p90/p95/p99
# par symbole. Rien d'inventé : sans échantillon -> dict honnête.
# La production n'est PAS modifiée : on commence par MESURER en continu.
# --------------------------------------------------------------------------- #
def friction_report(db, limit: int = 5000) -> dict:
    """Distribution réelle de la friction (slippage bps, fee % notional,
    latence ms) sur les fills persistés, globale et par symbole."""
    out = {"n_fills": 0, "slippage_bps": {}, "latency_ms": {},
           "fee_pct": None, "by_symbol": {}, "note": None}
    try:
        if db is None or not hasattr(db, "list_events"):
            out["note"] = "persistance indisponible"
            return out
        import numpy as np
        evs = db.list_events(event_type="paper_fill", limit=limit)
        rows = []
        for e in evs:
            try:
                d = json.loads(e.get("payload", "{}"))
                if d.get("slippage_bps") is not None:
                    rows.append(d)
            except Exception:
                continue
        out["n_fills"] = len(rows)
        if len(rows) < 10:
            out["note"] = (f"insuffisant : {len(rows)} fills persistés "
                           f"(≥ 10 pour une distribution) — la friction "
                           f"devient mesurable au fil des trades")
            return out
        slip = np.array([float(r["slippage_bps"]) for r in rows])
        lat = np.array([float(r["latency_ms"]) for r in rows
                        if r.get("latency_ms") is not None])
        notional = []
        for r in rows:
            try:
                q = float(r.get("qty") or 0.0)
                p = float(r.get("arrival") or 0.0)
                f = float(r.get("fee") or 0.0)
                if q > 0 and p > 0:
                    notional.append((f, q * p))
            except Exception:
                continue
        def _q(a, p):
            return round(float(np.percentile(a, p)), 2) if len(a) else None
        out["slippage_bps"] = {"p50": _q(slip, 50), "p90": _q(slip, 90),
                               "p95": _q(slip, 95), "p99": _q(slip, 99),
                               "max": round(float(slip.max()), 2)}
        out["latency_ms"] = {"p50": _q(lat, 50), "p95": _q(lat, 95)}
        if notional:
            fee_pct = 100.0 * sum(f for f, _ in notional) / sum(n for _, n in notional)
            out["fee_pct"] = round(float(fee_pct), 4)
        by_sym = {}
        for sym in sorted({r.get("symbol") for r in rows}):
            s = np.array([float(r["slippage_bps"]) for r in rows
                          if r.get("symbol") == sym])
            if len(s):
                by_sym[sym] = {"n": int(len(s)), "p50": _q(s, 50),
                               "p95": _q(s, 95), "p99": _q(s, 99)}
        out["by_symbol"] = by_sym
        out["note"] = ("friction mesurée sur les fills PERSISTÉS "
                       "(events paper_fill) — aucune donnée inventée")
    except Exception as e:
        out["note"] = f"indisponible ({e})"
    return out


# --------------------------------------------------------------------------- #
# PHASE 3 Cycle 5 — GATE DE FRICTION (décision opérateur, EURUSD micro-taille).
# Friction RÉELLE mesurée (890 fills) : EURUSD p95 157 bps vs AAPL/TSLA ~7 bps.
# Règle : un trade est refusé si le coût AR ATTENDU (frais réels 2×0,1 % +
# slippage p95 mesuré par symbole ×2) dépasse un seuil configuré
# (execution.max_expected_roundtrip_cost_pct, défaut 1,0 %). La production
# n'est PAS réglée sur le p95 : on refuse les QUEUES LOURDES mesurées.
# Sans mesure par symbole -> pas de blocage (aucun chiffre inventé).
# --------------------------------------------------------------------------- #
def expected_roundtrip_cost_pct(fee_pct: float, slippage_p95_bps: float | None,
                                default_bps: float = 0.0) -> float | None:
    """Coût AR attendu en % du notional : 2×frais + 2×slippage p95.
    Retourne None si le slippage p95 n'est pas mesuré (ni 0 ni inventé)."""
    if slippage_p95_bps is None:
        return None
    return 2.0 * fee_pct + 2.0 * (float(slippage_p95_bps) / 100.0)


def friction_gate_blocks(symbol: str, friction_cache: dict,
                         threshold_pct: float, fee_pct: float = 0.1) -> tuple:
    """
    True si le coût AR attendu (frais + slippage p95 mesuré du symbole)
    dépasse le seuil. Retourne (block, raison, coût_pct).
    Sans mesure p95 pour le symbole -> (False, None, None) : pas de blocage
    sans preuve.
    """
    p95 = (friction_cache or {}).get(symbol)
    cost = expected_roundtrip_cost_pct(fee_pct, p95)
    if cost is None:
        return False, None, None
    if cost > threshold_pct:
        return (True,
                f"friction: coût AR attendu {cost:.2f}% > seuil "
                f"{threshold_pct:.2f}% (slippage p95 mesuré {p95:.1f} bps)",
                round(cost, 4))
    return False, None, round(cost, 4)


def refresh_friction_cache(state: dict, db, max_age_sec: float = 600.0) -> dict:
    """Cache STATE['friction_p95_bps'] = {symbole: p95} rafraîchi au plus tous
    les max_age_sec (friction_report lit les events persistés). Jamais
    bloquant ; sans données -> cache vide (pas de blocage)."""
    try:
        cache = state.get("friction_p95_bps") or {}
        ts = state.get("friction_p95_ts") or 0.0
        if time.time() - ts < max_age_sec:
            return cache
        report = friction_report(db, limit=5000)
        out = {sym: d["p95"] for sym, d in (report.get("by_symbol") or {}).items()
               if d.get("p95") is not None}
        state["friction_p95_bps"] = out
        state["friction_p95_ts"] = time.time()
        state["friction_report_n"] = report.get("n_fills", 0)
        return out
    except Exception as e:
        logger.debug(f"refresh_friction_cache failed: {e}")
        return state.get("friction_p95_bps") or {}
