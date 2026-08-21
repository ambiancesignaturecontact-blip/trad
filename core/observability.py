"""
Suivi d'observation (P0-4 final_scale, P0-6 paper-validation) extrait de
main.py (LOT C, F3). Corps inchangés ; symboles partagés importés de main
de façon EXPLICITE (main est complet quand ce module est importé, en fin
de main.py).
"""

import logging

import json
import time

import numpy as np
from datetime import UTC
from main import (FINAL_SCALE_DOWNSAMPLE_SEC, FINAL_SCALE_MAX_SAMPLES, FINAL_SCALE_WINDOW_HOURS, STATE, db, settings)  # noqa: E402

logger = logging.getLogger("InstitutionalTradingBot")  # LOT C : même canal de logs que main


def _record_final_scale(symbol: str, final_scale: float, n_steps: int,
                        steps: list = None) -> None:
    """Accumule un échantillon (downsamplé à 1/min/symbole) de final_scale.
    Si les steps du pipeline sont fournis, le facteur LE PLUS RÉDUCTEUR
    (contrainte dominante — idée n°1 de l'audit) est mémorisé pour agréger
    « quel facteur bloque le trading » sur 24-48h."""
    try:
        now = time.time()
        last_ts = STATE.setdefault("final_scale_last_ts", {})
        if now - last_ts.get(symbol, 0.0) < FINAL_SCALE_DOWNSAMPLE_SEC:
            return
        last_ts[symbol] = now
        limiting = None
        if steps:
            best_name, best_val = None, 1.0
            for s in steps:
                if s.get("op") == "mul" and s.get("step") != "cumulative_floor":
                    v = float(s.get("value", 1.0))
                    if v < best_val:
                        best_name, best_val = s.get("step"), v
            if best_name is not None:
                limiting = {"factor": best_name, "value": best_val}
        samples = STATE.setdefault("final_scale_samples", [])
        samples.append({"ts": now, "symbol": symbol,
                        "final_scale": float(final_scale), "n_steps": int(n_steps),
                        "limit_factor": limiting["factor"] if limiting else None,
                        "limit_value": limiting["value"] if limiting else None})
        if len(samples) > FINAL_SCALE_MAX_SAMPLES:
            del samples[: len(samples) - FINAL_SCALE_MAX_SAMPLES]
        # P0-4 : persistance FRÉQUENTE (5 min) — si le process meurt entre deux
        # rapports 60 min, on ne perd que ~5 min de collecte au lieu de 60.
        if now - STATE.get("final_scale_last_persist", 0.0) >= 300.0:
            STATE["final_scale_last_persist"] = now
            _persist_final_scale_samples()
    except Exception:
        pass  # l'instrumentation ne doit jamais casser la boucle de trading


def _purge_final_scale_samples(max_age_sec: float) -> None:
    samples = STATE.get("final_scale_samples", [])
    if not samples:
        return
    cutoff = time.time() - max_age_sec
    # purge amortie : les échantillons sont horodatés de façon croissante
    while samples and samples[0]["ts"] < cutoff:
        samples.pop(0)


def _limiting_factor_stats() -> dict:
    """Agrège, sur l'échantillon accumulé, le facteur le plus souvent
    limitant (contrainte dominante) + sa réduction médiane — idée n°1 de
    l'audit : « c'est cash_reserve qui bloque 80 % du temps »."""
    samples = STATE.get("final_scale_samples", [])
    by_factor: dict = {}
    for s in samples:
        f = s.get("limit_factor")
        if not f:
            continue
        e = by_factor.setdefault(f, [])
        e.append(float(s.get("limit_value", 1.0)))
    if not by_factor:
        return {"n": 0, "top": []}
    rows = []
    for f, vals in by_factor.items():
        rows.append({"factor": f, "count": len(vals),
                     "pct_of_samples": round(100.0 * len(vals) / max(len(samples), 1), 1),
                     "median_value": round(float(np.median(vals)), 4)})
    rows.sort(key=lambda r: -r["count"])
    return {"n": len(samples), "top": rows[:5]}


def _final_scale_stats() -> dict:
    """Calcule p10/p50/p90/min/max sur l'échantillon accumulé (>= 5 points)."""
    samples = STATE.get("final_scale_samples", [])
    if len(samples) < 5:
        return None
    vals = np.array([s["final_scale"] for s in samples], dtype=float)
    p10, p50, p90 = np.percentile(vals, [10, 50, 90])
    return {
        "n": len(samples),
        "span_hours": round((samples[-1]["ts"] - samples[0]["ts"]) / 3600.0, 2),
        "p10": round(float(p10), 4),
        "p50": round(float(p50), 4),
        "p90": round(float(p90), 4),
        "min": round(float(vals.min()), 4),
        "max": round(float(vals.max()), 4),
    }


def _persist_final_scale_samples() -> None:
    """Persiste l'échantillon final_scale en DB pour survivre aux redémarrages
    (l'observation 24-48h de l'audit §2.1 ne doit pas repartir de zéro à chaque
    déploiement). Appelé à chaque rapport (toutes les 60 min) : perte max 60 min."""
    try:
        samples = STATE.get("final_scale_samples", [])
        db.save_setting("final_scale_samples_json", json.dumps(samples))
    except Exception:
        pass  # jamais bloquant


def _load_final_scale_samples() -> None:
    """Recharge l'échantillon persisté au démarrage (fenêtre 48h appliquée)."""
    try:
        raw = db.get_setting("final_scale_samples_json")
        if not raw:
            return
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            return
        samples = []
        for s in parsed:
            try:
                samples.append({"ts": float(s["ts"]), "symbol": str(s["symbol"]),
                                "final_scale": float(s["final_scale"]),
                                "n_steps": int(s.get("n_steps", 0))})
            except Exception:
                continue
        if samples:
            STATE["final_scale_samples"] = samples
            _purge_final_scale_samples(FINAL_SCALE_WINDOW_HOURS * 3600.0)
            logger.info(
                f"📊 FINAL_SCALE : {len(STATE['final_scale_samples'])} échantillons "
                f"chargés depuis la DB (observation persistée)."
            )
    except Exception as e:
        logger.warning(f"FINAL_SCALE : rechargement impossible ({e})")


def _signal_stats() -> dict:
    """Distribution de |final_signal| sur la fenêtre glissante (diagnostic
    conviction : si p50 est ~0.1, les signaux sont faibles juste au-dessus du
    seuil 0.08 — la question d'ajuster conviction se tranche sur CES données)."""
    sigs = [float(s) for s in STATE.get("recent_signals", []) if s is not None]
    if len(sigs) < 5:
        return {"n": len(sigs), "note": "échantillon insuffisant"}
    v = np.abs(np.array(sigs))
    return {
        "n": len(sigs),
        "abs_p10": round(float(np.percentile(v, 10)), 4),
        "abs_p50": round(float(np.percentile(v, 50)), 4),
        "abs_p90": round(float(np.percentile(v, 90)), 4),
        "threshold": STATE.get("conviction_threshold", 0.15),
        "entry_threshold": 0.08,
        "note": "p50 proche de 0.1 = signaux faibles juste au-dessus du seuil d'entrée.",
    }


def _final_scale_report() -> dict:
    """Purge + calcul + log périodique de la distribution de final_scale."""
    _purge_final_scale_samples(FINAL_SCALE_WINDOW_HOURS * 3600.0)
    stats = _final_scale_stats()
    if stats and stats["n"] >= 10:
        logger.info(
            f"📊 FINAL_SCALE distribution (n={stats['n']} sur {stats['span_hours']}h) : "
            f"p10={stats['p10']:.4f} p50={stats['p50']:.4f} p90={stats['p90']:.4f} "
            f"[min={stats['min']:.4f}, max={stats['max']:.4f}]"
        )
        if stats["p50"] < 0.20:
            logger.warning(
                "⚠️ FINAL_SCALE p50 < 20% : la chaîne de facteurs du pipeline de risque "
                "s'auto-amplifie (diagnostic audit §2.1) — le problème n'est pas le seuil "
                "de signal mais l'empilement de prudence."
            )
    else:
        logger.info(f"📊 FINAL_SCALE : échantillon insuffisant (n={stats['n'] if stats else 0}) — collecte en cours.")
    STATE["final_scale_stats"] = stats
    # idée n°1 audit : le facteur qui bloque le plus souvent le trading
    lim = _limiting_factor_stats()
    STATE["limiting_factor_stats"] = lim
    if lim and lim["top"]:
        t = lim["top"][0]
        top_str = ", ".join(
            "{} ({:.0f}%)".format(r["factor"], r["pct_of_samples"]) for r in lim["top"]
        )
        logger.info(
            "📊 FACTEUR LIMITANT (n={}) : '{}' contraint {:.1f}% des échantillons "
            "(valeur médiane {:.2f}). Top: {}".format(
                lim["n"], t["factor"], t["pct_of_samples"], t["median_value"], top_str)
        )
    _persist_final_scale_samples()
    return stats


def _mark_paper_validation_day() -> None:
    """Marque le jour UTC courant comme jour de paper-trading actif (le bot
    tourne réellement). Persisté en DB : la série survit aux redémarrages."""
    try:
        from datetime import datetime
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        days = json.loads(db.get_setting("paper_validation_days") or "[]")
        if not isinstance(days, list):
            days = []
        days = sorted(set(d for d in days if isinstance(d, str) and len(d) == 10))
        if today not in days:
            days.append(today)
            days.sort()
            db.save_setting("paper_validation_days", json.dumps(days))
        if not db.get_setting("paper_validation_start_ts"):
            db.save_setting("paper_validation_start_ts", str(time.time()))
    except Exception as e:
        logger.warning(f"paper validation mark failed: {e}")


def _paper_validation_stats() -> dict:
    """Jours actifs, série consécutive la plus récente, exigence, statut."""
    try:
        days = json.loads(db.get_setting("paper_validation_days") or "[]")
        start_ts = float(db.get_setting("paper_validation_start_ts") or 0.0)
    except Exception:
        days, start_ts = [], 0.0
    try:
        required = int(settings.get("autopilot", "min_paper_validation_days", 7))
    except Exception:
        required = 7
    days = sorted(set(d for d in days if isinstance(d, str) and len(d) == 10))

    # série consécutive la plus récente (fin = dernier jour enregistré)
    streak = 0
    if days:
        from datetime import datetime, timedelta
        try:
            cur = datetime.strptime(days[-1], "%Y-%m-%d").date()
            day_set = set(days)
            streak = 1
            while (cur - timedelta(days=1)).strftime("%Y-%m-%d") in day_set:
                streak += 1
                cur = cur - timedelta(days=1)
        except Exception:
            streak = 0

    return {
        "start_ts": start_ts,
        "active_days": len(days),
        "days": days,
        "latest_streak_days": streak,
        "required_days": required,
        "validated": streak >= required,
        "rule": "Le mode REAL exige une série CONTINUE de paper-trading daté "
                "(les jours où le bot n'a pas tourné ne comptent pas).",
    }
